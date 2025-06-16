import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.util import mine_hard_negatives
from torch.utils.data import DataLoader
from typing import List, Tuple
import random
from sklearn.model_selection import train_test_split
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLabelRanker:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/static-retrieval-mrl-en-v1",
                 cross_encoder_model_name: str = "answerdotai/ModernBERT-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化文档-标签排序器
        
        Args:
            embedding_model_name: 用于负样本挖掘的embedding模型
            cross_encoder_model_name: 用于微调的cross-encoder模型
            device: 设备选择
        """
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.device = device
        self.embedding_model = None
        self.cross_encoder = None
        
    def prepare_positive_samples(self, documents: List[str], label_lists: List[List[str]]) -> List[InputExample]:
        """
        准备正样本数据
        
        Args:
            documents: 文档列表
            label_lists: 对应的标签列表
            
        Returns:
            正样本InputExample列表
        """
        positive_samples = []
        
        for doc, labels in zip(documents, label_lists):
            for label in labels:
                # 创建正样本 (document, label, 1)
                positive_samples.append(InputExample(texts=[doc, label], label=1.0))
                
        logger.info(f"创建了 {len(positive_samples)} 个正样本")
        return positive_samples
    
    def mine_negative_samples(self, 
                            documents: List[str], 
                            label_lists: List[List[str]],
                            num_negatives: int = 5,
                            range_min: int = 10,
                            range_max: int = 100,
                            max_score: float = 0.8,
                            margin: float = 0.1,
                            batch_size: int = 4096) -> List[InputExample]:
        """
        使用sentence-transformer挖掘负样本
        
        Args:
            documents: 文档列表
            label_lists: 对应的标签列表
            num_negatives: 每个正样本对应的负样本数量
            range_min: 跳过前x个最相似的样本
            range_max: 只考虑前x个最相似的样本
            max_score: 只考虑相似度分数最多为x的样本
            margin: 查询与负样本的相似度应比查询-正样本相似度低x
            batch_size: 批处理大小
            
        Returns:
            包含正负样本的InputExample列表
        """
        # 初始化embedding模型
        if self.embedding_model is None:
            logger.info(f"加载embedding模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        
        # 准备正样本数据
        positive_samples = self.prepare_positive_samples(documents, label_lists)
        
        logger.info("开始挖掘负样本...")
        
        # 使用mine_hard_negatives挖掘负样本
        hard_dataset = mine_hard_negatives(
            positive_samples,
            self.embedding_model,
            num_negatives=num_negatives,
            range_min=range_min,
            range_max=range_max,
            max_score=max_score,
            margin=margin,
            sampling_strategy="top",
            batch_size=batch_size,
            output_format="labeled-pair",
            use_faiss=True
        )
        
        logger.info(f"挖掘完成，总共得到 {len(hard_dataset)} 个样本（正样本+负样本）")
        return hard_dataset
    
    def prepare_training_data(self, 
                            documents: List[str], 
                            label_lists: List[List[str]],
                            test_size: float = 0.2,
                            random_state: int = 42,
                            **mine_kwargs) -> Tuple[List[InputExample], List[InputExample]]:
        """
        准备训练和验证数据
        
        Args:
            documents: 文档列表
            label_lists: 对应的标签列表
            test_size: 测试集比例
            random_state: 随机种子
            **mine_kwargs: 负样本挖掘的参数
            
        Returns:
            训练集和验证集
        """
        # 挖掘负样本
        all_samples = self.mine_negative_samples(documents, label_lists, **mine_kwargs)
        
        # 划分训练集和验证集
        train_samples, val_samples = train_test_split(
            all_samples, 
            test_size=test_size, 
            random_state=random_state,
            stratify=[sample.label for sample in all_samples]
        )
        
        logger.info(f"训练集大小: {len(train_samples)}")
        logger.info(f"验证集大小: {len(val_samples)}")
        
        return train_samples, val_samples
    
    def train_cross_encoder(self,
                          train_samples: List[InputExample],
                          val_samples: List[InputExample],
                          output_path: str = "./cross-encoder-model",
                          num_epochs: int = 3,
                          train_batch_size: int = 16,
                          learning_rate: float = 2e-5,
                          warmup_steps: int = 100,
                          use_focal_loss: bool = False,
                          focal_alpha: float = 0.25,
                          focal_gamma: float = 2.0) -> CrossEncoder:
        """
        训练cross-encoder模型
        
        Args:
            train_samples: 训练样本
            val_samples: 验证样本
            output_path: 模型保存路径
            num_epochs: 训练轮数
            train_batch_size: 训练批大小
            learning_rate: 学习率
            warmup_steps: 预热步数
            use_focal_loss: 是否使用Focal Loss
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            
        Returns:
            训练好的CrossEncoder模型
        """
        # 初始化cross-encoder模型
        logger.info(f"初始化cross-encoder模型: {self.cross_encoder_model_name}")
        self.cross_encoder = CrossEncoder(
            self.cross_encoder_model_name, 
            num_labels=1,
            device=self.device
        )
        
        # 创建数据加载器
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        
        # 选择损失函数
        if use_focal_loss:
            logger.info("使用Focal Loss")
            train_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            logger.info("使用Binary Cross Entropy Loss")
            train_loss = losses.BinaryCrossEntropyLoss(self.cross_encoder)
        
        # 创建评估器
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            val_samples, 
            name='val-evaluator'
        )
        
        # 训练模型
        logger.info("开始训练模型...")
        self.cross_encoder.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=len(train_dataloader) // 2,  # 每半个epoch评估一次
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            optimizer_params={'lr': learning_rate}
        )
        
        logger.info(f"模型训练完成，保存到: {output_path}")
        return self.cross_encoder
    
    def rank_labels_for_document(self, 
                               document: str, 
                               candidate_labels: List[str],
                               model_path: str = None) -> List[Tuple[str, float]]:
        """
        为给定文档对标签列表进行排序
        
        Args:
            document: 文档文本
            candidate_labels: 候选标签列表
            model_path: 模型路径，如果None则使用当前模型
            
        Returns:
            排序后的标签和分数列表
        """
        if model_path:
            self.cross_encoder = CrossEncoder(model_path, device=self.device)
        
        if self.cross_encoder is None:
            raise ValueError("需要先训练模型或提供模型路径")
        
        # 创建文档-标签对
        pairs = [[document, label] for label in candidate_labels]
        
        # 预测分数
        scores = self.cross_encoder.predict(pairs)
        
        # 按分数排序
        labeled_scores = list(zip(candidate_labels, scores))
        labeled_scores.sort(key=lambda x: x[1], reverse=True)
        
        return labeled_scores


class FocalLoss:
    """
    Focal Loss实现，用于处理类别不平衡问题
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, predictions, labels):
        """
        计算Focal Loss
        
        Args:
            predictions: 模型预测值
            labels: 真实标签
            
        Returns:
            Focal Loss值
        """
        # 应用sigmoid激活函数
        probs = torch.sigmoid(predictions)
        
        # 计算交叉熵
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, labels, reduction='none'
        )
        
        # 计算pt
        pt = torch.where(labels == 1, probs, 1 - probs)
        
        # 计算alpha_t
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
