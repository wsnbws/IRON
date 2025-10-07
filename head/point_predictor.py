# Copyright (c) 2024. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .memory_attention import Attention
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import Mlp


class PointPredictor(nn.Module):
    """
    预测点提示模块：根据历史记忆和当前帧特征预测可通行区域的提示点
    
    设计思路：
    1. 初始化当前帧和历史帧的聚合token
    2. 使用3层self attention分别对不同帧进行语义聚合
    3. 进行一层最终聚合得到最终的聚合语义token
    4. 基于最终token预测置信度和点位置
    5. 返回聚合语义token、置信度和点位置
    """
    
    def __init__(
        self,
        current_dim: int = 256,
        memory_dim: int = 64,  
        num_points: int = 1,
        hidden_dim: int = 512,
        num_heads: int = 4,
        topk_size: int = 256,
        test_topk_size: int = 320,
        num_agg_tokens: int = 1, 
        num_self_attn_layers: int = 3,  
        hist_queue_length: int = 4, 
        image_size: tuple=(512, 512),
        test_image_size: tuple=(512, 640),
    ):
        """
        Args:
            current_dim: 当前帧特征通道数
            memory_dim: 历史记忆特征通道数
            num_points: 预测的点数量
            hidden_dim: MLP隐藏层维度
            num_heads: 注意力头数
            num_agg_tokens: 聚合token数量
            num_self_attn_layers: self attention层数
            k: 每帧的token数量，用于分离T和k
            hist_queue_length: 最大历史帧数量（队列长度）
        """
        super().__init__()
        self.current_dim = current_dim
        self.memory_dim = memory_dim
        self.num_points = num_points
        self.num_agg_tokens = num_agg_tokens
        self.num_self_attn_layers = num_self_attn_layers
        self.k = topk_size
        self.test_k = test_topk_size
        self.hist_queue_length = hist_queue_length
        self.image_size = image_size
        self.test_image_size = test_image_size
        
        self.unified_dim = hidden_dim // 2 
        self.current_agg_tokens = nn.Parameter(torch.randn(1, num_agg_tokens, self.unified_dim))
        self.memory_agg_tokens = nn.Parameter(torch.randn(hist_queue_length, num_agg_tokens, self.unified_dim))
        self.memory_proj = nn.Linear(memory_dim, self.unified_dim)
        self.current_proj = nn.Linear(current_dim, self.unified_dim)
        self.shared_self_attns = nn.ModuleList([
            Block(
                dim=self.unified_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                norm_layer=nn.LayerNorm
            ) for _ in range(num_self_attn_layers)
        ])
        
        self.norm1 = nn.LayerNorm(self.unified_dim * (num_agg_tokens + hist_queue_length*num_agg_tokens))
        self.final_mlp = Mlp(
            in_features=self.unified_dim * (num_agg_tokens + hist_queue_length*num_agg_tokens)  ,
            hidden_features=hidden_dim,
            out_features=self.unified_dim,
            act_layer=nn.GELU,
            drop=0.1
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(self.unified_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_points),
            nn.Sigmoid()
        )
        
        self.point_regressor = nn.Sequential(
            nn.Linear(self.unified_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_points * 2),  # num_points个点，每个点2个坐标
            nn.Sigmoid() 
        )
        
        self._init_weights()
    
    def _init_weights(self):

        nn.init.normal_(self.current_agg_tokens, std=0.02)
        nn.init.normal_(self.memory_agg_tokens, std=0.02) 
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def process_memory_features(
        self,
        mem_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mem_features: (S, B, C=64) 
            
        Returns:
            frame_agg_tokens: (B, T, num_agg_tokens, unified_dim) - content global tokens
        """

        S, B, C = mem_features.shape
        T = self.hist_queue_length
        k = self.k if self.training else self.test_k
        assert C == self.memory_dim, f"Expected memory_dim={self.memory_dim}, got {C}"
        assert S == T * k, f"S={S} should equal T*k={T}*{k}={T*k}"
        
        mem_features_reshaped = mem_features.transpose(0, 1).view(B, T, k, C)
        mem_features_proj = self.memory_proj(mem_features_reshaped).view(B * T, k, self.unified_dim)  # (B*T, k, unified_dim)
        
        frame_specific_tokens = self.memory_agg_tokens[:T, :, :]  # (T, num_agg_tokens, unified_dim)
        frame_tokens = frame_specific_tokens.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * T, self.num_agg_tokens, self.unified_dim)
        combined_frames = torch.cat([frame_tokens, mem_features_proj], dim=1)  # (B*T, num_agg_tokens + k, unified_dim)
        
        for self_attn_block in self.shared_self_attns:
            combined_frames = self_attn_block(combined_frames)  # (B*T, num_agg_tokens + k, unified_dim)
        frame_agg_tokens_flat = combined_frames[:, :self.num_agg_tokens, :]  # (B*T, num_agg_tokens, unified_dim)
        frame_agg_tokens = frame_agg_tokens_flat.view(B, T, self.num_agg_tokens, self.unified_dim)  # (B, T, num_agg_tokens, unified_dim)
        
        return frame_agg_tokens
    
    def process_current_features(
        self,
        cur_features: torch.Tensor,
        cur_pos_enc: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """      
        Args:
            cur_features: (B, C=256, H, W)
            cur_pos_enc: (B, C=256, H, W)
            
        Returns:
            current_agg_tokens: (B, num_agg_tokens, unified_dim) - content and position global tokens
        """
        B, C, H, W = cur_features.shape
        assert C == self.current_dim, f"Expected current_dim={self.current_dim}, got {C}"
        assert cur_pos_enc.shape == cur_features.shape, f"Position encoding shape {cur_pos_enc.shape} must match features shape {cur_features.shape}"
        
        combined_features = cur_features + cur_pos_enc  # (B, C=256, H, W)
        current_feature_tokens = combined_features.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C=256)
        current_feature_tokens = self.current_proj(current_feature_tokens)  # (B, H*W, unified_dim) 
        current_tokens = self.current_agg_tokens.expand(B, -1, -1)  # (B, num_agg_tokens, unified_dim)
        combined_current = torch.cat([current_tokens, current_feature_tokens], dim=1)  # (B, num_agg_tokens + H*W, unified_dim)
        
        for self_attn_block in self.shared_self_attns:
            combined_current = self_attn_block(combined_current)  # (B, num_agg_tokens + K, unified_dim)
        
        current_agg_tokens = combined_current[:, :self.num_agg_tokens, :]  # (B, num_agg_tokens, unified_dim)
        
        return current_agg_tokens
    
    def forward(
        self,
        cur_features: torch.Tensor,
        mem_features: torch.Tensor,
        cur_pos_enc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cur_features: (B, C=256, H, W) - 当前帧特征
            mem_features: (S, B, C=64) - 历史memory features，S=T*H*W或T*K
            memory_pos_enc: (S, B, C=64) - 历史memory位置编码
            cur_pos_enc: (B, C=256, H, W) - 当前帧位置编码
            image_size: (H_img, W_img) - 原始图像尺寸，用于将归一化坐标转换为像素坐标
            
        Returns:
            final_global_token: (B, unified_dim) - 最终的单个全局token
            confidence: (B, num_points) - 置信度分数, sigmoid [0, 1]
            points: (B, num_points, 2) - 预测的点坐标，范围[0, 1]或像素坐标
        """
        B = cur_features.shape[0]
        
        frame_agg_tokens = self.process_memory_features(mem_features)  # (B, T, num_agg_tokens, unified_dim)
        current_global_tokens = self.process_current_features(cur_features, cur_pos_enc)  # (B, num_agg_tokens, unified_dim)
        memory_global_tokens = frame_agg_tokens.flatten(1, 2)  # (B, T*num_agg_tokens, unified_dim)
        combined_tokens = torch.cat([current_global_tokens, memory_global_tokens], dim=1).flatten(1)  # (B, (num_agg_tokens + T*num_agg_tokens)*unified_dim)
        final_global_token = self.final_mlp(self.norm1(combined_tokens)) # (B, unified_dim)

        confidence = self.confidence_head(final_global_token)
        points_flat = self.point_regressor(final_global_token)  # (B, num_points * 2)
        points = points_flat.view(B, self.num_points, 2)  # (B, num_points, 2)
        image_size = self.test_image_size if self.training else self.image_size
        points = points * torch.tensor(list(self.image_size), device=points.device, dtype=points.dtype)
        
        return final_global_token, confidence, points  # (B, unified_dim), (B, num_points), (B, num_points, 2)


