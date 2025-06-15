# data_loader.py

from typing import List, Dict, Optional, Tuple
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

# 1. 加载文本数据
def load_texts(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f ]

# 2. 加载稀疏矩阵（X.tfidf.npz or Y.npz）
def load_sparse_matrix(path: str) -> csr_matrix:
    return load_npz(path)

# 3. 加载标签编号映射
def load_label_text_map(path: str) -> Dict[int, str]:
    """"
    Args:
        path: str
    Returns:
        mapping: Dict[int, str] where keys are label ids and values are label texts
    """
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for idx,val in enumerate(f):
            mapping[idx] = val.strip()
    return mapping

# 4. 编号转文本标签
def csr_id_to_text(id_mat: csr_matrix, label_map: Dict[int, str]) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Args:
        id_mat: csr_matrix where each row contains label ids (as column indices)
        label_map: dictionary from string label ID to label text

    Returns:
        Tuple:
            - List of lists of label texts for each row
            - List of lists of label indices (integers) for each row
    """
    all_label_texts = []
    all_label_indices = []

    for i in range(id_mat.shape[0]):
        start = id_mat.indptr[i]
        end = id_mat.indptr[i + 1]
        indices = id_mat.indices[start:end]  # 1D array of column indices (label ids)

        all_label_indices.append(indices.tolist())
        all_label_texts.append([label_map.get(idx, "Unknown") for idx in indices])

    return all_label_texts, all_label_indices
    
    

# 4. 文本 tokenizer 封装器
class TextTokenizer:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length

    def encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        token_type_ids = []

        for text in tqdm(texts, desc=f"Tokenizing with {self.model_name}"):
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids.append(encoding["input_ids"].squeeze(0))
            attention_mask.append(encoding["attention_mask"].squeeze(0))
            if "token_type_ids" in encoding:
                token_type_ids.append(encoding["token_type_ids"].squeeze(0))
            else:
                token_type_ids.append(torch.zeros_like(encoding["input_ids"].squeeze(0)))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "token_type_ids": torch.stack(token_type_ids)
        }

# 5. Dataset 类（支持稀疏标签）
class MultiLabelDataset(Dataset):
    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        label_matrix: csr_matrix,
        tfidf_matrix: Optional[csr_matrix] = None,
        max_labels: Optional[int] = None,
        padding_idx: int = -1
    ):
        self.inputs = encodings
        self.label_matrix = label_matrix
        self.tfidf_matrix = tfidf_matrix
        self.max_labels = max_labels
        self.padding_idx = padding_idx

    def __len__(self):
        return self.label_matrix.shape[0]

    def __getitem__(self, idx):
        x = {k: v[idx] for k, v in self.inputs.items()}
        y_row = self.label_matrix[idx]
        if self.max_labels:
            label_indices = y_row.indices
            label_values = y_row.data
            if len(label_indices) > self.max_labels:
                selected = torch.randperm(len(label_indices))[:self.max_labels]
                label_indices = label_indices[selected]
                label_values = label_values[selected]
            padded_idx = torch.full((self.max_labels,), self.padding_idx)
            padded_val = torch.zeros((self.max_labels,), dtype=torch.float32)
            padded_idx[:len(label_indices)] = torch.tensor(label_indices, dtype=torch.int64)
            padded_val[:len(label_values)] = torch.tensor(label_values, dtype=torch.float32)
            label_output = (padded_val, padded_idx)
        else:
            dense = torch.FloatTensor(y_row.toarray()).squeeze(0)
            label_output = dense

        if self.tfidf_matrix is not None:
            x_feat = torch.FloatTensor(self.tfidf_matrix[idx].toarray()).squeeze(0)
            return x, label_output, x_feat
        else:
            return x, label_output
