from typing import List, Tuple
from datasets import Dataset
from transformers import AutoTokenizer,PreTrainedTokenizerBase

def load_xmc_seq2seq_dataset(
    X_trn_text: List[str],
    Y_trn_text: List[str],
    X_val_text: List[str],
    Y_val_text: List[str],
    tokenizer_name: str,
    max_length: int = 128
) -> Tuple[Dataset, Dataset, PreTrainedTokenizerBase]:
    """
    加载并处理序列到序列任务的数据集

    Args:
        X_trn_text: 训练输入文本
        Y_trn_text: 训练输出文本
        X_val_text: 验证输入文本
        Y_val_text: 验证输出文本
        tokenizer_name: Tokenizer 名称或路径
        max_length: 最大 token 长度（用于 padding 和 truncation）

    Returns:
        Tokenized training dataset, validation dataset, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = Dataset.from_dict({'input': X_trn_text, 'output': Y_trn_text})
    val_data = Dataset.from_dict({'input': X_val_text, 'output': Y_val_text})

    def preprocess_function(batch):
        model_inputs = tokenizer(
            text=batch["input"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            text_target=batch["output"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_data.map(preprocess_function, batched=True)
    val_tokenized = val_data.map(preprocess_function, batched=True)

    return train_tokenized, val_tokenized, tokenizer
