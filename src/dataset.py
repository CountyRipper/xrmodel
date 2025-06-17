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

    train_data = Dataset.from_dict({'input': X_trn_text, 'output': Y_trn_text})
    val_data = Dataset.from_dict({'input': X_val_text, 'output': Y_val_text})

    def preprocess_function(batch):
        model_inputs = tokenizer(
            text=batch["input"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # 编码 target
        labels = tokenizer(
            batch["output"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        # 替换 pad_token 为 -100
        processed_labels = []
        for idx, label in enumerate(labels["input_ids"]):
            label = [tok if tok != tokenizer.pad_token_id else -100 for tok in label]
            if all(tok == -100 for tok in label):
                print(f"⚠️ Warning: label at index {idx} is all padding!")
            processed_labels.append(label)

        model_inputs["labels"] = processed_labels
        return model_inputs

    train_tokenized = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    val_tokenized = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)

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
