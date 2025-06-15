from typing import List, Optional, Tuple
from datasets import Dataset
from transformers import AutoTokenizer,PreTrainedTokenizerBase

def load_xmc_seq2seq_dataset(
    X_trn_text: List[str],
    Y_trn_text: List[str],
    X_val_text: List[str],
    Y_val_text: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    save_dir: Optional[str] = None
) -> Tuple[Dataset, Dataset]:
    """
    加载并处理序列到序列任务的数据集

    Args:
        X_trn_text: 训练输入文本
        Y_trn_text: 训练输出文本
        X_val_text: 验证输入文本
        Y_val_text: 验证输出文本
        tokenizer: Tokenizer 
        max_length: 最大 token 长度（用于 padding 和 truncation）
        save_dir: 保存处理后的数据集目录（可选）

    Returns:
        Tokenized training dataset, validation dataset, tokenizer
    """
    

    train_data = Dataset.from_dict({'input': X_trn_text, 'output': Y_trn_text})
    val_data = Dataset.from_dict({'input': X_val_text, 'output': Y_val_text})

    def preprocess_function(batch):
        model_inputs = tokenizer(
            text=batch["input"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
    
        # 编码 target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["output"],
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
    
        # 替换 padding token 为 -100，用于 loss 忽略
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_data.map(preprocess_function, batched=True)
    val_tokenized = val_data.map(preprocess_function, batched=True)

    # if output_dir exists and 
    if save_dir:
        train_tokenized.save_to_disk(f"{save_dir}/train_dataset")
        val_tokenized.save_to_disk(f"{save_dir}/val_dataset")
        print(f"Datasets saved to {save_dir}")

    return train_tokenized, val_tokenized

def load_xmc_seq2seq_dataset_from_disk(
    load_dir: str
) -> Tuple[Dataset, Dataset]:
    """
    从磁盘加载预处理好的序列到序列任务的数据集
    Args:
        load_dir: 数据集保存目录
        
    Returns:
        Tuple[Dataset, Dataset]: 训练集和验证集的 Dataset 对象
    """
    if load_dir is None:
        raise ValueError("load_dir is not validated.")
    # Load datasets from disk
    print(f"Loading datasets from {load_dir}")
    train_dataset = Dataset.load_from_disk(f"{load_dir}/train_dataset")
    val_dataset = Dataset.load_from_disk(f"{load_dir}/val_dataset")
    return train_dataset, val_dataset
