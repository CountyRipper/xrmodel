from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, PreTrainedModel,AutoTokenizer,Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,T5Tokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datasets import Dataset
from typing import Optional
from dataset import load_xmc_seq2seq_dataset
from xmcdata import csr_id_to_text, load_label_text_map, load_sparse_matrix, load_texts

@dataclass
class seq2seqParams:
    # seq2seq model config 
    model_name_or_path: str = 'bart'
    max_input_length: int = 128
    max_output_length: int = 64

    # training config
    num_train_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    use_fp16: bool = True
    logging_steps: int = 50
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    use_tensorboard: bool = True
    num_beams: int = 4
    @classmethod
    def load_config(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save_config(self, json_path: str):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)



class Seq2SeqModel(nn.Module):
    def __init__(self, config:seq2seqParams ):
        """
        初始化 HuggingFace 的 Seq2Seq 模型
        """
        super().__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path,cache_dir="./cache").to(self.device)
        if self.config.model_name_or_path.startswith('t5'):
            self.tokenizer: PreTrainedTokenizerBase = T5Tokenizer.from_pretrained(config.model_name_or_path,cache_dir="./cache", use_fast=True)
        else:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name_or_path,cache_dir="./cache")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def train_data_prepare(self, data_dir: Path|str)->Tuple[Dataset, Dataset]:
        """
        准备训练和验证数据集,也可在外部直接使用List[str]数据+load_xmc_seq2seq_dataset方法
        Args:
            data_dir: 数据目录，包含 X.trn.txt, Y.trn.txt, X.tst.txt, Y.tst.txt, output-items.txt
        Returns:
            Tuple[Dataset, Dataset]: 训练集和验证集的 Dataset 对象
        """
        data_dir = Path(data_dir)
        prefix = 'Summary this paragraph: '
        label_map = load_label_text_map(str(data_dir/   "output-items.txt"))

        X_trn_text = load_texts(str(data_dir/"X.trn.txt"))
        X_trn_text = [prefix + text for text in     X_trn_text]  # Add prefix to training texts
        Y_trn_text, _ = csr_id_to_text(load_sparse_matrix   (str(data_dir/"Y.trn.npz")), label_map)
        Y_trn_text = [",".join(y) for y in Y_trn_text]
        
        # 一定有test，所以不用判断是否存在
        X_val_text = load_texts(str(data_dir/"X.tst.txt"))
        X_val_text = [prefix + text for text in     X_val_text]  # Add prefix to validation texts
        Y_val_text, _ = csr_id_to_text(load_sparse_matrix   (str(data_dir/"Y.tst.npz")), label_map)
        #label formation
        Y_val_text = [",".join(y) for y in Y_val_text]
        
        train_dataset, val_dataset = load_xmc_seq2seq_dataset(X_trn_text, Y_trn_text, X_val_text, Y_val_text,self.tokenizer,max_length=self.config.max_input_length)

        return train_dataset, val_dataset

    def gen_train(
        self,
        train_dataset,
        val_dataset,
        output_dir
    )-> Tuple[Seq2SeqTrainer, str]:
        """
        训练模型
        Args:
            train_dataset: 训练集 Dataset 对象
            val_dataset: 验证集 Dataset 对象
            output_dir: 模型保存目录
        Returns:
            Tuple[Seq2SeqTrainer, str]: Seq2SeqTrainer 对象和输出目录
        """
        training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=self.config.learning_rate,
        per_device_train_batch_size=self.config.batch_size,
        per_device_eval_batch_size=self.config.batch_size,
        num_train_epochs=self.config.num_train_epochs,
        predict_with_generate=True,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=self.config.use_fp16,
        logging_steps=self.config.logging_steps,
        logging_dir=f"./{output_dir}/logs",
        warmup_ratio=self.config.warmup_ratio,
        weight_decay=self.config.weight_decay,
        load_best_model_at_end=True, # 早停效果？
        report_to=["tensorboard"] if self.config.use_tensorboard else []
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
        model=self.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=self.tokenizer,
        data_collator=data_collator,
        )
        print(f"Training model {self.config.model_name_or_path} with output directory {output_dir}")
        print(f"Training arguments: {self.config.__str__()}")
        trainer.train()
        trainer.save_model(output_dir)
        return trainer, output_dir


    def generate(self, input_ids, attention_mask=None, max_length=50,num_beams=4,**kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams = num_beams,
            **kwargs
        )

    def predict(self, inputs:List[str], batch_size=8, max_input_length=256,max_output_length=64,num_beams=4,output_path:Path|None =None)-> List[str]:
        """
        批量生成文本，自动处理 tokenization、GPU、tqdm 进度条
        Args:
            inputs: 输入文本列表 X.tst.txt: List[str]
            batch_size: 批量大小
            max_length: 生成文本的最大长度
            num_beams: beam search 的束宽
            output_path: 输出文件路径，如果为 None 则不保存到文件
        Returns:
            List[str]: 生成的文本列表
        """
        self.eval()
        self.cuda()
        results = []
        inputs = [item[:max_input_length] for item in inputs]  # 截断输入文本到最大长度
        if output_path:
            output_file = Path(output_path) / f"gen_pred_{str(self.config.model_name_or_path).split('/')[-1]}.txt"
            f=open(str(output_file),'w+')
            f.close()
        
        # clear file
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch = inputs[i:i+batch_size]
            encodings = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                output_ids = self.generate(**encodings, max_length=max_output_length,num_beams=num_beams)
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)
            if output_path:
                with open(str(output_file), 'a') as f:
                    for text in decoded:
                        f.write(text.strip(",") + "\n")
        return results

    
