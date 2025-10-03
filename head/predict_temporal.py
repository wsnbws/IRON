import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from typing import Optional, Tuple

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from .memory_encoder import MemoryEncoder
from .memory_attention import MemoryAttention
from .position_embed import PositionEmbeddingSine
from .history_queue import TemporalQueue
from .untils import LayerNorm2d
from .point_predictor import PointPredictor
from .loss import PointPredictionLoss
from .sam.prompt_encoder import PromptEncoder
from .sam.mask_decoder import MaskDecoder
from .sam.transformer import TwoWayTransformer


class PSPModule(nn.Module):
    """Pyramid Scene Parsing Module
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels  
        pool_scales (tuple): Pooling scales for pyramid pooling
    """
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Pyramid pooling branches
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, in_channels // len(pool_scales), kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels // len(pool_scales)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final conv to adjust channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * (in_channels // len(pool_scales)), 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """Forward pass of PSP module
        
        Args:
            x (Tensor): Input feature tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Enhanced feature tensor of shape (B, out_channels, H, W)
        """
        input_size = x.size()
        psp_outs = [x]
        
        for psp_module in self.psp_modules:
            psp_out = psp_module(x)
            psp_out = F.interpolate(
                psp_out, 
                size=input_size[2:], 
                mode='bilinear', 
                align_corners=False
            )
            psp_outs.append(psp_out)
        
        # Concatenate all pyramid features
        psp_outs = torch.cat(psp_outs, dim=1)
        
        # Final convolution to get desired output channels
        output = self.final_conv(psp_outs)
        
        return output

@HEADS.register_module()
class PredictiveTemporalUPerHead(BaseDecodeHead):
    """
    Predictive Temporal UPerHead that uses T-2 and T-1 frames to predict T frame,
    then fuses with current frame T for final prediction.
    """
    
    def __init__(self, **kwargs):
        # Pop custom streaming args before passing to parent
        self.streaming = bool(kwargs.pop('streaming', False))
        self.detach_every = int(kwargs.pop('detach_every', 0))
        self.history_length = int(kwargs.pop('history_length', 2))  # Configurable history length
        self.mask_ratio = int(kwargs.pop('mask_ratio', 8))
        
        # Top-K foreground selection configuration
        self.use_topk_memory = bool(kwargs.pop('use_topk_memory', False))  # 是否启用Top-K选择
        self.topk_memory_size = int(kwargs.pop('topk_memory_size', 256))   # Top-K的K值
        super(PredictiveTemporalUPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # ===== Streaming (true online) inference/training state (no new learnable params) =====
        self.temporal_queue = TemporalQueue(
            history_length=self.history_length, 
            streaming=self.streaming
        )
        self.temporal_time_test_queue = None

        self.last_m1_logits = None
        self._build_fpn_module()
        self.memory_attention = MemoryAttention()
        self.memory_encoder = MemoryEncoder(total_stride=self.mask_ratio)
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000
        )
        
        # PSP module for enhanced semantic perception
        self.psp_module = PSPModule(
            in_channels=self.channels,
            out_channels=self.channels,
            pool_scales=(1, 2, 3, 6)
        )
        # time embedding: simple MLP mapping scalar timestamp -> channel embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channels//4)
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            LayerNorm2d(self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )

        self.hist_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.channels, self.channels, kernel_size=2, stride=2
            ),
            nn.GELU(),
            nn.Conv2d(self.channels, self.num_classes, kernel_size=3, padding=1)
        )

        # Point prediction modules
        self.point_predictor = PointPredictor(
            current_dim=self.channels,
            memory_dim=64,
            num_points=1,
            hidden_dim=512,
            num_heads=4,
            use_topk_features=True,
            topk_size=64,
        )
        self.point_loss = PointPredictionLoss(
            cls_weight=1.0,
            reg_weight=1.0,
            normalize_by_image_size=True,
            min_area_ratio=0.0,
        )
        self.point_prompt_threshold = 0.5

        # Prompt encoder and mask decoder will be lazily initialized
        self.prompt_encoder: Optional[PromptEncoder] = None
        self.mask_decoder: Optional[MaskDecoder] = None
        self.prompt_two_way_transformer: Optional[TwoWayTransformer] = None

        # Debug hooks for downstream use
        self.last_point_logits: Optional[torch.Tensor] = None
        self.last_point_probs: Optional[torch.Tensor] = None
        self.last_pred_points: Optional[torch.Tensor] = None
        self.last_prompt_masks: Optional[torch.Tensor] = None

   
    def _build_fpn_module(self):
     # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
    # ===== Helpers for streaming mode =====
    def _fpn_forward_single(self, inputs):
        """UPer forward for a single frame batch. Returns (B, C, H, W) features after fpn_bottleneck."""
        inputs = self._transform_inputs(inputs)

        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        return fpn_outs

    def _get_memory(self, cur_features, t = 0, timestamps: torch.Tensor = None):
        B, C, H, W = cur_features.shape 
        # Ensure queue allocation
        mask_shape = (self.num_classes - 1, H * self.mask_ratio, W * self.mask_ratio)
        self.temporal_queue.ensure_allocation(cur_features, mask_shape)
        # Reset at sequence start
        if int(t) == 0:
            self.temporal_queue.reset_state(all_batch=True)

        # Read historical frames from queue
        # Clone to decouple from underlying queue storage to avoid inplace version conflicts
        historical_frames, historical_masks = self.temporal_queue.get_history_frames() # (T, B, C, H, W)
        
        # Reshape for batch processing: (T*B, C, H, W)
        batch_frames = historical_frames.view(-1, *historical_frames.shape[2:])
        batch_masks = historical_masks.view(-1, *historical_masks.shape[2:])

        # Vectorized time embedding processing
        timestamps = timestamps.to(cur_features.device, dtype=cur_features.dtype)
        # normalize UNIX timestamps per-sample to [0, 1]
        tmin = timestamps.min(dim=1, keepdim=True)[0]
        tmax = timestamps.max(dim=1, keepdim=True)[0]
        denom = (tmax - tmin).clamp_min(1e-6)
        timestamps = (timestamps - tmin) / denom
        
        historical_timestamps = timestamps[:, :-1].transpose(0, 1)                                    # (num_historical, B)
        batch_timestamps = historical_timestamps.reshape(-1, 1)                                       # (num_historical * B, 1)
        batch_time_enc = self.time_mlp(batch_timestamps).unsqueeze(2).unsqueeze(3).expand(-1, -1, *historical_frames.shape[3:]) # (num_historical * B, C_time_embed, H, W)

        # Single forward pass for all historical frames
        batch_memory_output = self.memory_encoder(batch_frames, batch_masks)
        mem_features = batch_memory_output["vision_features"]  # (T*B, C, H, W)
        batch_pos_enc = batch_memory_output["vision_pos_enc"]    # (T*B, C, H, W)
        batch_pos_enc = batch_pos_enc + batch_time_enc
        
        # 根据配置决定是否使用Top-K前景选择
        if self.use_topk_memory:
            # 使用Top-K选择前景memory tokens
            if self.training:
                selected_memory, selected_pos_enc = self._select_topk_foreground_memory(
                    mem_features, batch_pos_enc, batch_masks, k=self.topk_memory_size
                )
            else:
                selected_memory, selected_pos_enc = self._select_topk_foreground_memory(
                    mem_features, batch_pos_enc, batch_masks, k=320
                )
            # 转换为attention所需的格式: (S, B, C)
            k_tokens = selected_memory.shape[0] // (len(historical_frames) * B)  # 每个样本的token数
            memory_enc = selected_pos_enc.view(len(historical_frames), B, k_tokens, -1).permute(0, 2, 1, 3).flatten(0, 1)  # (T*k, B, C)
            mem_features = selected_memory.view(len(historical_frames), B, k_tokens, -1).permute(0, 2, 1, 3).flatten(0, 1)  # (T*k, B, C)
        else:
            # 使用原始的全图memory tokens (向后兼容)
            batch_pos_enc = batch_pos_enc.view(-1, B, *batch_pos_enc.shape[1:]) #(T, B, C, H, W)
            memory_enc = batch_pos_enc.permute(0, 3, 4, 1, 2).flatten(0, 2) #(T*H*W, B,C)
            
            mem_features = mem_features.view(-1, B, *mem_features.shape[1:])
            mem_features = mem_features.permute(0, 3, 4, 1, 2).flatten(0, 2)
        return memory_enc, mem_features

    def _select_topk_foreground_memory(self, mem_features, batch_pos_enc, batch_masks, k=256):
        """
        基于masks选择Top-K前景memory tokens (完全向量化版本)
        
        Args:
            mem_features: (T*B, C, H, W) - 历史帧特征
            batch_pos_enc: (T*B, C, H, W) - 位置编码
            batch_masks: (T*B, num_classes-1, H_mask, W_mask) - 分割掩码
            k: 每个样本选择的token数量
            
        Returns:
            selected_memory: (T*B*k, C) - 选择的memory特征
            selected_pos_enc: (T*B*k, C) - 对应的位置编码
        """
        TB, C, H, W = mem_features.shape
        
        # 将mask下采样到特征图尺寸并求和得到前景概率
        masks_resized = F.interpolate(
            batch_masks.sum(dim=1, keepdim=True),  # (T*B, 1, H_mask, W_mask)
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)  # (T*B, H, W)
        
        # 展平空间维度
        masks_flat = masks_resized.view(TB, H*W)  # (T*B, H*W)
        mem_flat = mem_features.view(TB, C, H*W)  # (T*B, C, H*W)
        pos_flat = batch_pos_enc.view(TB, C, H*W)  # (T*B, C, H*W)
        
        # 向量化的Top-K选择
        _, topk_indices = torch.topk(masks_flat, k=min(k, H*W), dim=1)  # (T*B, k)
        
        # 使用gather选择对应的特征和位置编码
        # 扩展indices以匹配特征维度
        topk_indices_expanded = topk_indices.unsqueeze(1).expand(-1, C, -1)  # (T*B, C, k)
        
        selected_memory = torch.gather(mem_flat, 2, topk_indices_expanded)  # (T*B, C, k)
        selected_pos_enc = torch.gather(pos_flat, 2, topk_indices_expanded)  # (T*B, C, k)
        
        # 转换为所需格式
        selected_memory = selected_memory.transpose(1, 2).reshape(TB*k, C)  # (T*B*k, C)
        selected_pos_enc = selected_pos_enc.transpose(1, 2).reshape(TB*k, C)  # (T*B*k, C)
        
        return selected_memory, selected_pos_enc

    def _forward_stream_batch(self, inputs, t: int, timestamps: torch.Tensor = None, **kargs):
        """Streaming forward for a batch of single-frame features using per-slot memory.

        Args:
            inputs: list of tensors for a single time step, each (B, C, H, W)
            t: current time index; reset state when t == 0
            timestamps: (B, 3) float tensor for [t-2, t-1, t] (optional)

        Returns:
            M1: (B, num_classes, H, W)
            final_output: (B, num_classes, H, W)
        """
        # Compute current frame features (B, C, H, W)
        fpn_outs = self._fpn_forward_single(inputs)
        if len(fpn_outs) > 1:
            cur_features = fpn_outs[-1]  # (B, 768, 32, 32)
        
        # Apply PSP module for enhanced semantic perception
        cur_features = self.psp_module(cur_features)  # (B, C, H, W)
        
        B, C, H, W = cur_features.shape 
        
        # cur_features = self._corrupt_current_features(cur_features)
        cur_pos_enc = self.pos_embed(cur_features)  # (B, C, H, W)
        cur_features_seq = cur_features.flatten(2).permute(2, 0, 1)  # (S, B, C)
        cur_pos_enc_seq = cur_pos_enc.flatten(2).permute(2, 0, 1)
    
        memory_enc, mem_features = self._get_memory(cur_features, t, timestamps)
        
        fus_feat, hist_feat = self.memory_attention(cur_features_seq, mem_features, cur_pos_enc_seq, memory_enc, spatial_shape=(H, W), **kargs)
        fus_feat = fus_feat.permute(1, 2, 0).reshape(B, C, H, W)  # (bs, c, h, w)
        hist_feat = hist_feat.permute(1, 2, 0).reshape(B, C, H, W)

        # Point prediction and prompt generation
        has_point_logits, pred_points = self.point_predictor(
            cur_features=cur_features,
            mem_features=mem_features,
            memory_pos_enc=memory_enc,
            image_size=(512, 512),
        )
        sparse_prompt_embeddings, dense_prompt_embeddings = self.prompt_encoder(
            pred_points, has_point_logits, confidence_is_logit=True
        )
        sparse_prompt_embeddings = sparse_prompt_embeddings.to(fus_feat.device, fus_feat.dtype)
        dense_prompt_embeddings = dense_prompt_embeddings.to(fus_feat.device, fus_feat.dtype)

        masks = self.mask_decoder(
            image_embeddings=fus_feat,
            image_pe=cur_pos_enc,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        self.last_prompt_masks = masks.detach()

        # Update queues: keep pure current features and final outputs
        self.temporal_queue.push(cur_features, final_output[:,-1:,:,:])

        return final_output, hist_output, has_point_logits, pred_points, masks, use_point_mask

    def _corrupt_current_features(self, features: torch.Tensor) -> torch.Tensor:

        if not self.training:
            return features
            
        corruption_prob = 0.8
        if torch.rand(1).item() > corruption_prob:
            return features
            
        corrupted = features.clone()
        B, C, H, W = corrupted.shape
        
        dropout_ratio = 0.5
        mask = torch.rand(B, 1, H, W, device=features.device) > dropout_ratio
        corrupted = corrupted * mask.float()

        return corrupted

    def forward(self, inputs):
        return None
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Test-time forward. Supports streaming with single-frame inputs.

        If ``self.streaming`` is True, expects per-step single-frame features
        and uses ``img_metas`` to determine sequence start for reset.
        """
        metas = img_metas
        do_reset = any(bool((mb.get('is_seq_start', False))) for mb in metas if isinstance(mb, dict))
        basename = os.path.basename(metas[0]['filename'])
        cur_timestamp = float(os.path.splitext(basename)[0].split("_")[-1])
        group_index = int(metas[0]['group_index'])

        t_val = 0 if do_reset else 1
        if group_index == 0:
            self.temporal_time_test_queue = [cur_timestamp] * (self.history_length + 1)
        else:
            self.temporal_time_test_queue.pop(0)
            self.temporal_time_test_queue.append(cur_timestamp)
            if group_index + 1 < self.history_length + 1:
                for i in range(self.history_length - group_index):
                    self.temporal_time_test_queue[i] = cur_timestamp

        ts_tensor = torch.tensor([self.temporal_time_test_queue], dtype=torch.float32)
        final_output, _ = self._forward_stream_batch(inputs, t_val, timestamps=ts_tensor, basename = basename)
        return final_output
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, t=0, timestamps: torch.Tensor = None,):
        """Forward function for training with two heads (M1 and final).

        Computes losses for both M1 and final outputs using the base ``losses``.
        """
        # Streaming path: inputs contain only current frame features (B, ...)
        final_output, hist_output, has_point_logits, pred_points, _, _ = self._forward_stream_batch(inputs, t, timestamps=timestamps)
        loss_final = self.losses(final_output, gt_semantic_seg)
        loss_hist = self.losses(hist_output, gt_semantic_seg)
        loss_point_total, loss_point_cls, loss_point_reg, point_targets = self.point_loss(
            pred_has_point=has_point_logits,
            pred_points=pred_points,
            gt_semantic_seg=gt_semantic_seg,
            target_class=1,
        )
        losses = {}
        for key, value in loss_final.items():
            losses[f'{key}_final'] = value
        for key, value in loss_hist.items():
            losses[f'{key}_hist'] = value
        losses['loss_point'] = loss_point_total
        losses['loss_point_cls'] = loss_point_cls
        losses['loss_point_reg'] = loss_point_reg
        losses['point_targets'] = point_targets.mean()
        return losses
