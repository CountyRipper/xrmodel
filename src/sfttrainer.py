from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import bitsandbytes as bnb

@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"  # 可替换为其他模型
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
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
    prompt_template: str = "Summarize this paragraph by keyphrases: {document}\n"
    max_new_tokens: int = 100  # 生成时的最大新token数


class Sft_trainer:
    def __init__(self, config:ModelConfig,data_dir:str):
        self.config = config
        self.data_dir = data_dir
        print("Initializing SFTTrainer with model:", self.config.model_name)
        print("Data directory:", self.data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.config = config
        

    def load_dataset(self):
        print("Loading dataset from: ", self.data_dir)
        return load_dataset("csv", data_files={"train": self.data_dir + "train.csv", "test": self.data_dir + "test.csv"})
    def _preprocess(self,example):
        example["labels"] = example["labels"].split(",")
        return example
    def _format_prompt(self, example):
        prompt = self.config.prompt_template+ "\nAnswer:"
        response = ", ".join(example["labels"])
        return {"prompt": prompt, "completion": response}
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
    
    def train(self,  dataset=None):
        if dataset is None:
            dataset = self.load_dataset()
        # 预处理数据集
        dataset = dataset.map(self._preprocess)
        dataset = dataset.map(self._format_prompt)
        
        # 标准化字段名
        dataset = dataset.rename_columns({"prompt": "document", "completion": "completion"})
        
        # 数据 collator
        collator = DataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer, instruction_template="Summarize this paragraph by keyphrases: {document}\n", response_template="Answer:")
        self.setup_lora()  # 设置LoRA配置
        self.setup_quantization()  # 设置量化配置
        print("setting trainer args")
        # 训练参数
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            logging_steps=self.config.logging_steps,
            warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.quantization_type == "fp16",
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            num_train_epochs=self.config.num_epochs,
            learning_rate=2e-5,
            bf16=False,  # 如果你用 A100 or 4090 等
            gradient_checkpointing=True,
            dataloader_drop_last=False,
            logging_dir=f"./{self.config.output_dir}/logs",
            report_to=["tensorboard"] if self.config.use_tensorboard else [],
            remove_unused_columns=False,
        )
        
        # 初始化 SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            args=args,
            data_collator=collator,
            max_seq_length=512,
            formatting_func=lambda x: f"{x['document']}{x['completion']}"
        )
        
        # 启动训练
        trainer.train()
        
        # 保存模型和分词器
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")
    def load_trained_model(self, model_path: str):
        """
        加载训练好的模型和分词器
        :param model_dir: 模型保存的目录
        """
        #basemodel
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
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text[len(prompt):].strip()
    def batch_generate(self, test_dataset, prompt: str, max_new_tokens: int = 128, output_dir: str = None):
        """
        批量生成答案
        :param test_dataset: 测试数据集
        :param prompt: 提示词模板
        :param max_new_tokens: 最大生成的token数
        :return: 生成的答案列表
        """
        
        # === 批量推理 ===
        results = []
        if output_dir:
            # 刷新pred.txt文件
            f = open(output_dir + self.config.model_name+"_pred.txt", "w", encoding="utf-8")
            f.close()
        for example in tqdm(test_dataset, desc="Generating"):
            prompt = prompt.replace("{document}", example["document"])
            prediction = self.generate_answer(prompt, max_new_tokens=max_new_tokens)
            if output_dir:
                with open(output_dir + self.config.model_name+"_pred.txt", "a", encoding="utf-8") as f:
                    f.write(f"{prediction}\n")
            results.append({
                "prompt": prompt,
                "prediction": prediction
            })
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT for LLM")
    parser.add_argument('--dataset_name', type=str, default='eurlex-4k', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.2-3B-Instruct', help='Name of the model')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = args.model_name
    data_dir = f"xmc-base/{dataset_name}/"

    modelconfig = ModelConfig(
        model_name=model_name,
        output_dir=f"./output/{dataset_name}/{model_name}",
        prompt_template="Summarize this paragraph by keyphrases: {document}\n",
        max_new_tokens=128,  # 生成的最大新令牌数
        batch_size=4,
        learning_rate=2e-4,
        warmup_steps=100,
        warmup_ratio=0.1,
        num_epochs=4,
        use_tensorboard=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=False,  # 是否使用量化
        quantization_type="int4"  # 可选: "int4", "int8", "fp16", "fp32"
    )


    trainer = Sft_trainer(modelconfig, data_dir=data_dir)
    print("Starting SFT training...")
    # 1. 加载数据集
    dataset = trainer.load_dataset()
    print("Dataset loaded successfully")
    
    trainer.train(dataset=dataset)
    print("Training completed successfully")

    # 预测
    print("Starting batch generation...")
    trainer.batch_generate(
        test_dataset=dataset["test"],
        prompt=modelconfig.prompt_template,
        max_new_tokens=modelconfig.max_new_tokens,
        output_dir=data_dir
    )
    

    
