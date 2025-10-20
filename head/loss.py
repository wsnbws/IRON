# Copyright (c) 2024. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import cv2
from head.flag import get_task_state
class otdr_loss(nn.Module):
    """
    Unified loss for video segmentation with point prediction and mask generation.
    
    Combines three loss components:
    1. Point existence classification loss (binary cross-entropy)
    2. Point coordinate regression loss (L2 distance)
    3. Mask segmentation loss (binary cross-entropy with logits)
    
    This unified approach ensures consistent training objectives and simplifies
    loss computation for end-to-end video segmentation models.
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        seg_weight: float = 1.0,
        normalize_by_image_size: bool = True,
        min_area_ratio: float = 0.0,
        ignore_index: int = 255,
    ) -> None:
        """
        Initialize unified segmentation loss.
        
        Args:
            cls_weight (float): Weight for point classification loss. Default: 1.0
            reg_weight (float): Weight for point regression loss. Default: 1.0
            seg_weight (float): Weight for mask segmentation loss. Default: 1.0
            normalize_by_image_size (bool): Whether to normalize point distances by image diagonal. Default: True
            min_area_ratio (float): Minimum foreground area ratio for valid point targets. Default: 0.0
            ignore_index (int): Index to ignore in mask loss computation. Default: 255
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.seg_weight = seg_weight
        self.normalize_by_image_size = normalize_by_image_size
        self.min_area_ratio = min_area_ratio
        self.ignore_index = ignore_index
    
    def compute_center_from_mask(
        self,
        gt_masks: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mask centers using distance transform for point supervision.
        
        Args:
            gt_masks (Tensor): Ground truth masks (B, H, W) with integer class labels
            target_class (int): Class index to treat as foreground. Default: 1
        
        Returns:
            tuple[Tensor, Tensor, Tensor]: Point targets and validity information
                - centers (Tensor): Mask centers (B, 2) with (x, y) pixel coordinates
                - valid_mask (Tensor): Valid foreground presence flags (B,) as boolean
                - areas (Tensor): Foreground pixel counts per sample (B,) as float
        """
        B, H, W = gt_masks.shape
        device = gt_masks.device
        
        # Convert to numpy for OpenCV distance transform
        target_mask = (gt_masks == target_class).cpu().numpy().astype(np.uint8)
        
        centers_list = []
        valid_list = []
        area_list = []
        
        for i in range(B):
            mask = target_mask[i]  # (H, W)
            
            # Add background border around mask to prevent edge bias in distance transform
            # This ensures that masks touching image boundaries are properly handled
            mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0) # set border to background
            mask_area = mask.sum()
            area_list.append(float(mask_area))

            if mask_area == 0:
                # No foreground pixels found
                centers_list.append([0.0, 0.0])
                valid_list.append(False)
                continue
            
            # Compute distance transform to find mask center
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # Find location with maximum distance (center of largest inscribed circle)
            max_loc = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
            center_y, center_x = max_loc
            
            centers_list.append([float(center_x), float(center_y)])
            valid_list.append(True)
        
        # Convert back to tensors
        centers = torch.tensor(centers_list, dtype=torch.float32, device=device)
        valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=device)
        areas = torch.tensor(area_list, dtype=torch.float32, device=device)
        
        return centers, valid_mask, areas
    
    def compute_point_losses(
        self,
        pred_has_point: torch.Tensor,
        pred_points: torch.Tensor,
        gt_semantic_seg: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute point prediction losses (classification + regression).
        
        Args:
            pred_has_point (Tensor): Point existence probabilities (B,) or (B, 1) - already sigmoid-activated
            pred_points (Tensor): Predicted point coordinates (B, 1, 2) in pixel space
            gt_semantic_seg (Tensor): Ground truth masks (B, 1, H, W) with class labels
            target_class (int): Foreground class index. Default: 1
        
        Returns:
            tuple[Tensor, Tensor, Tensor]: Point prediction losses
                - cls_loss (Tensor): Binary classification loss for point existence
                - reg_loss (Tensor): L2 regression loss for point coordinates
                - target_has_point (Tensor): Ground truth point existence flags (B,)
        """
        # Prepare tensors
        pred_has_point = pred_has_point.squeeze(-1)  # (B,)
        gt_masks = gt_semantic_seg.squeeze(1)  # (B, H, W)
        H, W = gt_masks.shape[1:]
        
        # Compute ground truth point targets
        gt_centers, valid_mask, areas = self.compute_center_from_mask(gt_masks, target_class)
        
        # Determine valid point targets based on minimum area threshold
        min_pixels = self.min_area_ratio * float(H * W)
        target_has_point = ((areas >= min_pixels) & valid_mask).float()  # (B,)
        
        # Classification loss: point existence prediction
        # Note: pred_has_point is already sigmoid-activated, so use binary_cross_entropy instead of binary_cross_entropy_with_logits
        cls_loss = F.binary_cross_entropy(pred_has_point, target_has_point)
        
        # Regression loss: point coordinate prediction (only for positive samples)
        positive_mask = target_has_point > 0.0
        if positive_mask.any():
            valid_pred = pred_points[positive_mask].squeeze(1)  # (N_pos, 2)
            valid_gt = gt_centers[positive_mask]  # (N_pos, 2)
            
            # Normalize coordinates by image size before computing loss
            # x normalized by W, y normalized by H
            scale_vec = torch.tensor([W, H], dtype=valid_pred.dtype, device=valid_pred.device)
            valid_pred_norm = valid_pred / scale_vec
            valid_gt_norm = valid_gt / scale_vec

            # Compute L2 distance in normalized coordinate space
            distance = torch.norm(valid_pred_norm - valid_gt_norm, p=2, dim=1)  # (N_pos,)
            
            reg_loss = distance.mean()
        else:
            # No positive samples: zero regression loss
            reg_loss = pred_points.sum() * 0.0
        
        return cls_loss, reg_loss, target_has_point
    
    def compute_segmentation_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        target_class: int = 1
    ) -> torch.Tensor:
        """
        Compute binary segmentation loss for mask prediction.
        
        Args:
            pred_masks (Tensor): Predicted mask logits (B, 1, H_pred, W_pred) before sigmoid
            gt_masks (Tensor): Ground truth masks (B, 1, H_gt, W_gt) or (B, H_gt, W_gt) with class labels
            target_class (int): Class index to treat as foreground. Default: 1
        
        Returns:
            Tensor: Binary cross-entropy segmentation loss
        """
        # Ensure consistent tensor shapes
        if gt_masks.dim() == 3:  # (B, H, W)
            gt_masks = gt_masks.unsqueeze(1)  # (B, 1, H, W)
        
        # Get target spatial dimensions from ground truth
        _, _, H_gt, W_gt = gt_masks.shape
        _, _, H_pred, W_pred = pred_masks.shape
        
        # Upsample predicted masks to match ground truth resolution
        if (H_pred, W_pred) != (H_gt, W_gt):
            pred_masks = F.interpolate(
                pred_masks,  # (B, 1, H_pred, W_pred)
                size=(H_gt, W_gt),  # Target size from ground truth
                mode='bilinear',  # Bilinear interpolation for smooth upsampling
                align_corners=False  # Don't align corners for better interpolation
            )  # Output: (B, 1, H_gt, W_gt)
        
        # Create binary target mask (foreground vs background)
        binary_gt = (gt_masks == target_class).float()  # (B, 1, H_gt, W_gt)
        
        # Flatten tensors for loss computation
        pred_flat = pred_masks.view(-1)  # (B*H_gt*W_gt,)
        target_flat = binary_gt.view(-1)  # (B*H_gt*W_gt,)
        
        # Compute binary cross-entropy loss with logits
        seg_loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat, reduction='mean')
        return seg_loss
    
    def forward(
        self,
        pred_masks,
        gt_semantic_seg: torch.Tensor,
        pred_has_point: torch.Tensor = None,
        pred_points: torch.Tensor = None,
        target_class: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute unified loss combining point prediction and mask segmentation.
        
        Args:
            pred_has_point (Tensor): Point existence probabilities (B,) or (B, 1) - already sigmoid-activated
            pred_points (Tensor): Predicted point coordinates (B, 1, 2)
            pred_masks (Tensor): Predicted mask logits (B, 1, H, W)
            gt_semantic_seg (Tensor): Ground truth masks (B, 1, H, W) with class labels
            target_class (int): Foreground class index. Default: 1
        
        Returns:
            tuple[Tensor, dict]: Combined loss and detailed loss components
                - total_loss (Tensor): Weighted sum of all loss components
                - loss_dict (dict): Individual loss components for monitoring:
                    - 'loss_point_cls': Point classification loss
                    - 'loss_point_reg': Point regression loss  
                    - 'loss_mask_seg': Mask segmentation loss
                    - 'point_targets': Mean number of valid point targets
        """
        # Compute point prediction losses
        # cls_loss, reg_loss, target_has_point = self.compute_point_losses(
        #     pred_has_point, pred_points, gt_semantic_seg, target_class
        # )
        
        # Compute mask segmentation loss
        if gt_semantic_seg.dim() == 3:  # (B, H, W)
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)  # (B, 1, H, W)
            
        if isinstance(pred_masks, list) or isinstance(pred_masks, tuple):
            mask_coarse, mask_mid, mask_fine = pred_masks
            fine_loss = self.compute_segmentation_loss(
                mask_fine, gt_semantic_seg, target_class
            )
            mid_loss = self.compute_segmentation_loss(
                mask_mid, F.interpolate(gt_semantic_seg.float(), scale_factor=0.25, mode='nearest').long(), target_class
            )
            coarse_loss = self.compute_segmentation_loss(
                mask_coarse, F.interpolate(gt_semantic_seg.float(), scale_factor=0.125, mode='nearest').long(), target_class
            )
        else:   
            seg_loss = self.compute_segmentation_loss(
                pred_masks, gt_semantic_seg, target_class
            )
        
        # Prepare detailed loss information
        loss_dict = {
            # 'loss_point_cls': cls_loss * self.cls_weight,
            # 'loss_point_reg': reg_loss * self.reg_weight,
            'mask_coarse_loss': coarse_loss,
            'mask_mid_loss': mid_loss,
            'mask_fine_loss': fine_loss,
        }
        
        return loss_dict