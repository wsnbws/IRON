# Copyright (c) 2024. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PointPredictor(nn.Module):
    """
    预测点提示模块：根据历史记忆和当前帧特征预测可通行区域的提示点
    
    设计思路：
    1. 将历史记忆聚合成全局上下文token
    2. 与当前帧特征交互，预测点的位置
    3. 输出归一化的点坐标 (B, num_points, 2)
    """
    
    def __init__(
        self,
        current_dim: int = 256,  # 当前帧特征维度
        memory_dim: int = 64,    # 历史记忆特征维度
        num_points: int = 5,
        hidden_dim: int = 512,
        num_heads: int = 4,
        use_topk_features: bool = True,
        topk_size: int = 64,
    ):
        """
        Args:
            current_dim: 当前帧特征通道数
            memory_dim: 历史记忆特征通道数
            num_points: 预测的点数量
            hidden_dim: MLP隐藏层维度
            num_heads: 注意力头数
            use_topk_features: 是否使用Top-K池化来聚合当前帧特征
            topk_size: Top-K的K值
        """
        super().__init__()
        self.current_dim = current_dim
        self.memory_dim = memory_dim
        self.num_points = num_points
        self.use_topk_features = use_topk_features
        self.topk_size = topk_size
        
        # 统一特征维度
        self.unified_dim = hidden_dim // 2  # 使用hidden_dim的一半作为统一维度
        
        # ===== 历史记忆聚合 =====
        # 先将memory_dim投影到unified_dim
        self.memory_proj = nn.Linear(memory_dim, self.unified_dim)
        
        # 使用cross-attention将历史memory聚合成固定数量的tokens
        self.memory_aggregation = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 可学习的查询tokens用于聚合历史记忆
        self.memory_query = nn.Parameter(torch.randn(1, num_points, self.unified_dim))
        
        # ===== 当前帧特征提取 =====
        # 先将current_dim投影到unified_dim
        self.current_proj = nn.Linear(current_dim, self.unified_dim)
        
        # 对当前帧进行全局池化或Top-K池化
        if use_topk_features:
            # 使用可学习的重要性评分来选择Top-K特征
            self.importance_score = nn.Sequential(
                nn.Conv2d(current_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # ===== 点位置预测 =====
        # 将聚合的历史记忆 + 当前帧特征 融合后预测点坐标
        self.point_fusion = nn.Sequential(
            nn.Linear(self.unified_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 预测点存在性（has_point 二分类）
        self.cls_head = nn.Linear(hidden_dim, 1)

        # 预测每个点的(x, y)坐标，范围[0, 1]
        self.point_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # 归一化到[0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化memory_query
        nn.init.normal_(self.memory_query, std=0.02)
        
        # 初始化MLP
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def aggregate_memory(
        self,
        mem_features: torch.Tensor,
        memory_pos_enc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        聚合历史记忆特征
        
        Args:
            mem_features: (S, B, C=64) - 历史memory tokens，S=T*H*W或T*K
            memory_pos_enc: (S, B, C=64) - 对应的位置编码
            
        Returns:
            aggregated_memory: (B, num_points, unified_dim) - 聚合后的记忆tokens
        """
        S, B, C = mem_features.shape
        assert C == self.memory_dim, f"Expected memory_dim={self.memory_dim}, got {C}"
        
        # 投影到统一维度
        mem_features_bt = mem_features.transpose(0, 1)  # (B, S, C=64)
        mem_features_proj = self.memory_proj(mem_features_bt)  # (B, S, unified_dim)
        
        # 位置编码也投影
        if memory_pos_enc is not None:
            memory_pos_bt = memory_pos_enc.transpose(0, 1)  # (B, S, C=64)
            memory_pos_proj = self.memory_proj(memory_pos_bt)  # (B, S, unified_dim)
        else:
            memory_pos_proj = 0
        
        # 扩展query到batch维度
        query = self.memory_query.expand(B, -1, -1)  # (B, num_points, unified_dim)
        
        # 使用cross-attention聚合历史记忆
        aggregated_memory, _ = self.memory_aggregation(
            query=query,
            key=mem_features_proj + memory_pos_proj,
            value=mem_features_proj
        )  # (B, num_points, unified_dim)
        
        return aggregated_memory
    
    def extract_current_features(
        self,
        cur_features: torch.Tensor
    ) -> torch.Tensor:
        """
        提取当前帧的全局特征
        
        Args:
            cur_features: (B, C=256, H, W) - 当前帧特征
            
        Returns:
            current_global: (B, num_points, unified_dim) - 当前帧全局特征
        """
        B, C, H, W = cur_features.shape
        assert C == self.current_dim, f"Expected current_dim={self.current_dim}, got {C}"
        
        if self.use_topk_features:
            # 使用Top-K池化
            importance = self.importance_score(cur_features)  # (B, 1, H, W)
            importance_flat = importance.view(B, H * W)  # (B, H*W)
            
            # 选择Top-K个最重要的位置
            _, topk_indices = torch.topk(
                importance_flat, 
                k=min(self.topk_size, H * W), 
                dim=1
            )  # (B, K)
            
            # 提取对应位置的特征
            cur_features_flat = cur_features.view(B, C, H * W)  # (B, C=256, H*W)
            topk_indices_expanded = topk_indices.unsqueeze(1).expand(-1, C, -1)  # (B, C, K)
            topk_features = torch.gather(cur_features_flat, 2, topk_indices_expanded)  # (B, C=256, K)
            
            # 平均池化得到全局特征
            current_global = topk_features.mean(dim=2)  # (B, C=256)
        else:
            # 使用全局平均池化
            current_global = F.adaptive_avg_pool2d(cur_features, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C=256)
        
        # 投影到统一维度
        current_global = self.current_proj(current_global)  # (B, unified_dim)
        
        # 扩展到num_points维度，让每个点都能访问全局特征
        current_global = current_global.unsqueeze(1).expand(-1, self.num_points, -1)  # (B, num_points, unified_dim)
        
        return current_global
    
    def forward(
        self,
        cur_features: torch.Tensor,
        mem_features: torch.Tensor,
        memory_pos_enc: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        预测点提示
        
        Args:
            cur_features: (B, C=256, H, W) - 当前帧特征
            mem_features: (S, B, C=64) - 历史memory features，S=T*H*W或T*K
            memory_pos_enc: (S, B, C=64) - 历史memory位置编码
            image_size: (H_img, W_img) - 原始图像尺寸，用于将归一化坐标转换为像素坐标
            
        Returns:
            has_point_logits: (B,) - 点存在性的logits
            points: (B, num_points, 2) - 预测的点坐标，范围[0, 1]或像素坐标
        """
        B = cur_features.shape[0]
        
        aggregated_memory = self.aggregate_memory(mem_features, memory_pos_enc)  # (B, num_points, unified_dim)
        
        current_global = self.extract_current_features(cur_features)  # (B, num_points, unified_dim)
        
        fused_features = torch.cat([aggregated_memory, current_global], dim=-1)  # (B, num_points, 2*unified_dim)
        fused_features = self.point_fusion(fused_features)  # (B, num_points, hidden_dim)
        
        has_point_logits = self.cls_head(fused_features.mean(dim=1)).squeeze(-1)  # (B,)

        points = self.point_regressor(fused_features)  # (B, num_points, 2), 范围[0, 1]
        
        if image_size is not None:
            H_img, W_img = image_size
            points = points * torch.tensor([W_img, H_img], device=points.device, dtype=points.dtype)
        
        return has_point_logits, points


