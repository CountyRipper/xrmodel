import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Set
from tqdm import tqdm

class LabelReplacementSystem:
    """
    使用embedding模型进行标签替换的系统
    支持List[List[str]]输入格式，自动去重
    """
    
    def __init__(self, label_map: List[str], model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化标签替换系统
        
        Args:
            label_map: 标签库，包含所有候选标签
            model_name: 使用的embedding模型名称,可替换，如'Qwen3-Embedding-0.6B'等
        """
        # # 去重标签库
        # self.label_map = list(set(label_map))
        # self.label_map_set = set(self.label_map)
        self.label_map = label_map
        # label_map不需要去重
        self.model = SentenceTransformer(model_name)
        
        # 创建标签到索引的映射
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_map)}
        
        print(f"标签库大小: {len(self.label_map)}")
        print(f"使用模型: {model_name}")
        
        # 预计算标签库embeddings
        self._precompute_library_embeddings()
    
    def _precompute_library_embeddings(self):
        """预计算标签库中所有标签的embeddings"""
        print("计算标签库embeddings...")
        self.library_embeddings = self.model.encode(
            self.label_map, 
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True  # 直接归一化
        )
        print("标签库embeddings计算完成!")
    
    def _deduplicate_label_list(self, label_list: List[str]) -> List[str]:
        """
        去除单个label list中的重复项,保持原始顺序
        """
        # seen = set()
        # result = []
        # for label in label_list:
        #     if label not in seen:
        #         result.append(label)
        #         seen.add(label)
        # 直接使用dict.fromkeys保持顺序并去重
        return list(dict.fromkeys(label_list))  # 使用dict保持顺序并去重
        
    
    def _find_best_replacement(self, target_label: str, existing_labels: Set[str]) -> tuple:
        """
        为目标标签找到最佳替换
        
        Args:
            target_label: 需要替换的标签
            existing_labels: 当前label list中已存在的标签
            
        Returns:
            (替换标签, 相似度分数) 或 (None, 0.0) 如果无法替换
        """
        # 计算目标标签的embedding
        target_embedding = self.model.encode(
            [target_label], 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # 计算与标签库的余弦相似度
        # 也可以使用 sentence_transformers的util.cos_sim计算,返回一个tensor
        similarities = torch.mm(target_embedding, self.library_embeddings.t()).squeeze(0)
        
        # 创建掩码，屏蔽已存在的标签
        mask = torch.ones(len(self.label_map), dtype=torch.bool, device=similarities.device)
        for existing_label in existing_labels:
            if existing_label in self.label_to_idx:
                mask[self.label_to_idx[existing_label]] = False
        
        # 应用掩码
        masked_similarities = similarities.clone()
        masked_similarities[~mask] = float('-inf')
        
        # 找到最高分数的标签
        if mask.sum() == 0:  # 没有可用标签
            return None, 0.0
        
        best_idx = torch.argmax(masked_similarities).item()
        best_label = self.label_map[best_idx]
        best_score = similarities[best_idx].item()
        
        return best_label, best_score
    
    def replace_single_list(self, label_list: List[str]) -> Dict:
        """
        处理单个标签列表
        
        Args:
            label_list: 原始标签列表
            
        Returns:
            处理结果字典
        """
        # 步骤0: 去重
        deduplicated_list = self._deduplicate_label_list(label_list)
        
        result = {
            'original': label_list,
            'deduplicated': deduplicated_list,
            'final': [],
            'replacements': {},  # 原标签 -> (新标签, 分数)
            'kept_original': [],  # 保持不变的标签
            'stats': {
                'original_count': len(label_list),
                'after_dedup': len(deduplicated_list),
                'replacements_made': 0,
                'kept_from_library': 0
            }
        }
        
        current_labels = set()  # 跟踪当前结果中的标签
        
        for label in deduplicated_list:
            if label in self.label_map:
                # 标签在库中，直接保留
                result['final'].append(label)
                result['kept_original'].append(label)
                current_labels.add(label)
                result['stats']['kept_from_library'] += 1
            else:
                # 标签不在库中，需要替换
                replacement, score = self._find_best_replacement(label, current_labels)
                
                if replacement is not None:
                    result['final'].append(replacement)
                    result['replacements'][label] = (replacement, score)
                    current_labels.add(replacement)
                    result['stats']['replacements_made'] += 1
                else:
                    # 无法找到替换，保持原标签（极少发生）
                    result['final'].append(label)
                    current_labels.add(label)
        
        return result
    
    def replace_batch(self, label_lists: List[List[str]], show_progress: bool = True) -> List[Dict]:
        """
        批量处理标签列表
        
        Args:
            label_lists: List[List[str]] 格式的标签列表
            show_progress: 是否显示进度条
            
        Returns:
            处理结果列表
            获取处理后的标签列表 final_labels = [result['final'] for result in results]
        """
        results = []
        
        iterator = tqdm(label_lists, desc="处理标签列表", disable=not show_progress)
        
        for i, label_list in enumerate(iterator):
            result = self.replace_single_list(label_list)
            result['index'] = i
            results.append(result)
        
        return results
    
    def get_summary_stats(self, results: List[Dict]) -> Dict:
        """
        获取汇总统计信息
        
        Args:
            results: 处理结果列表
            
        Returns:
            统计信息字典
        """
        total_lists = len(results)
        total_original = sum(r['stats']['original_count'] for r in results)
        total_after_dedup = sum(r['stats']['after_dedup'] for r in results)
        total_replacements = sum(r['stats']['replacements_made'] for r in results)
        total_kept = sum(r['stats']['kept_from_library'] for r in results)
        
        # 计算替换分数统计
        all_scores = []
        for result in results:
            for _, (_, score) in result['replacements'].items():
                all_scores.append(score)
        
        dedup_reduction = ((total_original - total_after_dedup) / total_original * 100) if total_original > 0 else 0
        replacement_rate = (total_replacements / total_after_dedup * 100) if total_after_dedup > 0 else 0
        
        return {
            'total_lists_processed': total_lists,
            'total_labels_original': total_original,
            'total_labels_after_dedup': total_after_dedup,
            'deduplication_reduction': f"{dedup_reduction:.1f}%",
            'total_replacements_made': total_replacements,
            'total_kept_from_library': total_kept,
            'replacement_rate': f"{replacement_rate:.1f}%",
            'avg_similarity_score': f"{np.mean(all_scores):.3f}" if all_scores else "N/A",
            'min_similarity_score': f"{np.min(all_scores):.3f}" if all_scores else "N/A",
            'max_similarity_score': f"{np.max(all_scores):.3f}" if all_scores else "N/A"
        }