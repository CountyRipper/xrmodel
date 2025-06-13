from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, PreTrainedModel
import torch
import torch.nn as nn
from tqdm import tqdm

class Seq2SeqModel(nn.Module):
    def __init__(self, model_name_or_path: str):
        """
        初始化 HuggingFace 的 Seq2Seq 模型
        """
        super().__init__()
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids, attention_mask=None, max_length=50, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )

    def predict(self, tokenizer: PreTrainedTokenizerBase, inputs, batch_size=8, max_length=50):
        """
        批量生成文本，自动处理 tokenization、GPU、tqdm 进度条
        """
        self.eval()
        self.cuda()
        results = []

        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch = inputs[i:i+batch_size]
            encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                output_ids = self.generate(**encodings, max_length=max_length)
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)

        return results

    from typing import Optional

    def save(self, save_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.model.save_pretrained(save_path)
        if tokenizer:
            tokenizer.save_pretrained(save_path)

    @classmethod
    def load(cls, model_path: str):
        return cls(model_path)
