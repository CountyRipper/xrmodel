#用于data的提前处理，和data.sh一起使用
'''
@ author: Yui
@ date: 2025/06/20
@ description: 预处理数据集，主要用于将txt文件转换为csv可以直接用dataset处理的模式
'''
import csv
from typing import List, Optional, Dict, Tuple
from scipy.sparse import load_npz, csr_matrix

def load_texts(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f ]

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


def save_to_csv(documents: List[str], labels: List[List[str]], output_file: str, label_sep: str = ","):
    """
    将文档和标签写入CSV文件，labels为多个单词，用逗号或自定义分隔符分隔

    Args:
        documents: 文本列表
        labels: 标签列表，每个元素是一个 label list（多标签）
        output_file: 保存路径
        label_sep: 标签之间的分隔符（默认英文逗号）
    """
    assert len(documents) == len(labels), "文档与标签数量不一致"

    with open(output_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["document", "labels"])  # header
        for doc, lbls in zip(documents, labels):
            label_str = label_sep.join(lbls)
            writer.writerow([doc, label_str])

def preprocess_text_file(data_dir:str=None, label_sep: str = ","):
    """
    预处理文本文件，将其转换为CSV格式，适用于多标签分类任务

    Args:
        data_dir: 数据目录，包含train.txt和val.txt
        label_sep: 标签之间的分隔符（默认英文逗号）
    """
    if data_dir is None:
        raise ValueError("data_dir must be specified")
    print("preprocess_text_file: data_dir:", data_dir)
    #data_dir = f"xmc-base/{dataset_name}"
    #laod label map
    label_map = load_label_text_map(data_dir + "/output-items.txt")
    # training dataset
    train_text_list = load_texts(data_dir + "/X.trn.txt")
    train_label_feat = load_npz(data_dir + "/Y.trn.npz")
    train_label_list,train_label_num = csr_id_to_text(train_label_feat, label_map)
    train_label_text_list = [label_sep.join(y) for y in train_label_list]

    # validation dataset
    test_text_list = load_texts(data_dir + "/X.tst.txt")
    test_label_feat = load_npz(data_dir + "/Y.tst.npz")
    test_label_list, test_label_num = csr_id_to_text(test_label_feat, label_map)
    test_label_text_list = [label_sep.join(y) for y in test_label_list]

    # 输出处理结果
    print("save_to_csv: train_text_list length:", len(train_text_list))
    print("save_to_csv: train_label_list length:", len(train_label_list))
    print("save_dir:", data_dir+"/train.csv")
    save_to_csv(train_text_list, train_label_list, data_dir + "/train.csv", label_sep)

    print("save_to_csv: test_text_list length:", len(test_text_list))
    print("save_to_csv: test_label_list length:", len(test_label_list))
    print("save_dir:", data_dir+"/test.csv")
    save_to_csv(test_text_list, test_label_list, data_dir + "/test.csv", label_sep)
    
    return {
        "train_text_list": train_text_list,
        "train_label_list": train_label_list,
        "test_text_list": test_text_list,
        "test_label_list": test_label_list,
        "label_map": label_map
    }

    