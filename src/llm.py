import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,DataCollatorWithPadding,
    BitsAndBytesConfig
)
from datasets import Dataset, load_dataset, Features, Sequence, Value
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import bitsandbytes as bnb
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"  # 可替换为其他模型
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    output_dir: str = "./results"
    use_tensorboard: bool = True  # 是否使用TensorBoard记录训练日志
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 量化配置
    use_quantization: bool = False
    quantization_type: str = "int4"  # "int4", "int8", "fp16", "fp32"

    # template
    prompt_template: str = ""
    max_new_tokens: int = 100  # 生成时的最大新token数

class DataProcessor:
    """数据处理类"""
    def __init__(self, tokenizer, max_length_input: int = 256, max_length_output: int = 128, max_length: int = 512,prompt_template: str = None, res_template: str = None):
        self.tokenizer = tokenizer
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output
        self.max_length = max_length_input + max_length_output
        
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = (
                "Summarize the following document with keyphrases:\n"
                "Document: {document}\n"
            )
        if res_template:
            self.res_template = res_template
        else:
            self.res_template = (
                "Summary of this paragraph by keyphrases: {keyphrases}"
            )
    
    def _prepare_data_from_lists(self, documents: List[str], keyphrases: List[str]) -> Dataset:
        """从列表数据准备训练数据集"""
        data = []
        for doc, kp in zip(documents, keyphrases):
            input_text = self.prompt_template.format(document=doc)
            output_text = self.res_template.format(keyphrases=kp)
            
            data.append({
                "input_text": input_text,
                "output_text": output_text
            })
        return Dataset.from_list(data)
    
    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        input_texts = examples["input_text"]
        output_texts = examples["output_text"]
        if len(input_texts) != len(output_texts):
            raise ValueError("Input texts and output texts must have the same length.")
        input_ids_batch = []
        labels_batch = []
        self.tokenizer.padding_side = "right"  # 确保padding在右侧
        for input, output in zip(input_texts, output_texts):
            # 不使用tensor,使用list
            input_encode = self.tokenizer(
                input,
                padding="max_length",
                truncation=True,
                max_length=self.max_length_input,
                add_special_tokens=False
            )
            output_encode = self.tokenizer(
                output,
                truncation=True,
                padding="max_length",
                max_length=self.max_length_output,
                add_special_tokens=False
            )
            input_ids = input_encode["input_ids"]
            output_ids = output_encode["input_ids"]
            #手动拼接input_ids和output_ids
            full_input_ids = input_ids + output_ids
            # 生成labels
            labels = [-100] * len(input_ids) + [token if token!= self.tokenizer.pad_token_id else -100 for token in output_ids]
            input_ids_batch.append(full_input_ids)
            labels_batch.append(labels)
        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            #"attention_mask": [[1] * len(ids) for ids in input_ids_batch]  # 添加attention_mask
        }

    def prepare_dataset(self, documents: List[str], keyphrases: List[str], num_proc: int =4,
                        batch_size: int = 8) -> Dataset:
        mydataset = self._prepare_data_from_lists(documents, keyphrases)
        # 使用map函数进行tokenize
        tokenized_dataset = mydataset.map(
            self._tokenize_function,
            batched=True,
            batch_size=batch_size,  # 可以根据内存大小调整
            num_proc= num_proc,  # 并行处理的进程数
            remove_columns=mydataset.column_names,
            desc="Tokenizing dataset"
        )
        features = Features({
            "input_ids": Sequence(Value(dtype="int32")),
            "labels": Sequence(Value(dtype="int32")),
            #"attention_mask": Sequence(Value(dtype="int32"))
            })

        tokenized_dataset = tokenized_dataset.cast(features)
        return tokenized_dataset
    def save_dataset(self, dataset: Dataset, save_path: str):
        """保存数据集到指定路径"""
        dataset.save_to_disk(save_path)
        print(f"Dataset saved to {save_path}")
    def load_dataset(self, load_path: str) -> Dataset:
        """从指定路径加载数据集"""
        dataset = Dataset.load_from_disk(load_path)
        print(f"Dataset loaded from {load_path}")
        return dataset
    
class PeftTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # 👈 添加 **kwargs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss

class LLMTrainer:
    """LLM训练器类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        # 加载tokenizer        print(f"Loading model: {self.config.__str__()}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(self.tokenizer, self.config.max_length)
        print("Model and tokenizer loaded successfully!")
        
    def setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """设置量化配置"""
        if not self.config.use_quantization:
            return None
            
        if self.config.quantization_type == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization_type == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            return None
    
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        print(f"Loading model: {self.config.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(self.tokenizer, self.config.max_length)
        
        print("Model and tokenizer loaded successfully!")
    
    def setup_lora(self):
        """设置LoRA配置"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 根据模型调整
        )
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        #self.model.config.use_cache = False  # LoRA模型不需要cache
        print("LoRA configuration applied successfully!")
    
    def prepare_training_args(self) -> TrainingArguments:
        """准备训练参数"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,

            save_strategy="epoch",
            eval_strategy="epoch",       
            load_best_model_at_end=True,
            learning_rate=self.config.learning_rate,
            fp16=self.config.quantization_type == "fp16",
            bf16=False,
            gradient_checkpointing=True,
            dataloader_drop_last=False,
            logging_dir=f"./{self.config.output_dir}/logs",
            report_to=["tensorboard"] if self.config.use_tensorboard else [],
            save_total_limit=1,
            remove_unused_columns= False
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, output_dir: str = None):
        """训练模型"""
        print("Starting training...")
        
        # 数据整理器
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config.max_length,
        )
        # data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=self.tokenizer,
        #     mlm=False,
        # )
        
        # 训练参数
        training_args = self.prepare_training_args()
        self.model.config.use_cache = False  # 必须显式关闭
        # 创建trainer
        trainer = PeftTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        print("self.model.training: ",self.model.training)
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print("Training completed!")
    
    def save_model(self, save_path: str):
        """保存模型"""
        if save_path is None:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        else:
            ValueError("Save path cannot be None.")
        print(f"Model saved to {save_path}")
    
    def load_trained_model(self, model_path: str):
        """加载训练好的模型"""
        print(f"Loading trained model from {model_path}")
        
        # 加载base model
        quantization_config = self.setup_quantization()
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "torch_dtype": torch.float16 if self.config.quantization_type == "fp16" else torch.float32,
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        base_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Trained model loaded successfully!")

class KeyphrasePredictor:
    """关键词预测器"""
    
    def __init__(self, trainer: LLMTrainer):
        self.trainer = trainer
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Please load the model and tokenizer before prediction.")
        self.data_processor = trainer.data_processor
    
    def predict(self, documents: List[str], max_new_tokens: int = 100) -> List[str]:
        """预测关键词"""
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for doc in documents:
                # 构建输入prompt
                input_text = self.data_processor.prompt_template.format(document=doc)
                
                # tokenize
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.trainer.config.max_length - max_new_tokens
                ).to(self.model.device)
                
                # 生成
                with torch.autocast("cuda", enabled=self.trainer.config.quantization_type == "fp16"):
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # 解码输出
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                predictions.append(generated_text)
        
        return predictions
    
    def batch_predict(self, documents: List[str], batch_size: int = 4, max_new_tokens: int = 100,data_dir:str = None) -> List[str]:
        """批量预测（带进度条）"""
        predictions = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Batch Prediction"):
            batch_docs = documents[i:i + batch_size]
            batch_predictions = self.predict(batch_docs, max_new_tokens)
            predictions.extend(batch_predictions)
        
        if data_dir:
            output_file = f"{data_dir}/predictions.txt"
            with open(output_file, 'w') as f:
                for pred in predictions:
                    f.write(pred + "\n")
            print(f"Predictions saved to {output_file}")
        return predictions
    
    
