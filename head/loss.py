# Copyright (c) 2024. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import cv2


class PointPredictionLoss(nn.Module):
    """Binary existence + point regression loss for prompt generation."""

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        normalize_by_image_size: bool = True,
        min_area_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.normalize_by_image_size = normalize_by_image_size
        self.min_area_ratio = min_area_ratio
    
    def compute_center_from_mask(
        self,
        gt_masks: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mask centers via distance transform.

        Args:
            gt_masks: (B, H, W) tensor of integer class labels.
            target_class: class id treated as foreground.

        Returns:
            centers: (B, 2) tensor with pixel coordinates (x, y).
            valid_mask: (B,) bool tensor marking valid foreground presence.
            areas: (B,) tensor with foreground pixel counts.
        """
        B, H, W = gt_masks.shape
        device = gt_masks.device
        
        target_mask = (gt_masks == target_class).cpu().numpy().astype(np.uint8)
        
        centers_list = []
        valid_list = []
        area_list = []
        
        for i in range(B):
            mask = target_mask[i]
            mask_area = mask.sum()
            area_list.append(float(mask_area))

            if mask_area == 0:
                centers_list.append([0.0, 0.0])
                valid_list.append(False)
                continue
            
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            max_loc = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
            center_y, center_x = max_loc
            
            centers_list.append([float(center_x), float(center_y)])
            valid_list.append(True)
        
        centers = torch.tensor(centers_list, dtype=torch.float32, device=device)
        valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=device)
        areas = torch.tensor(area_list, dtype=torch.float32, device=device)
        
        return centers, valid_mask, areas
    
    def forward(
        self,
        pred_has_point: torch.Tensor,
        pred_points: torch.Tensor,
        gt_semantic_seg: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_has_point: (B,) or (B, 1) logits for point existence.
            pred_points:  (B, 1, 2) predicted (x, y) pixels.
            gt_semantic_seg: (B, 1, H, W) integer mask labels.
            target_class: foreground class id.

        Returns:
            total_loss, cls_loss, reg_loss, target_has_point (B,).
        """
        pred_has_point = pred_has_point.squeeze(-1)
        gt_masks = gt_semantic_seg.squeeze(1)
        H, W = gt_masks.shape[1:]
        
        gt_centers, valid_mask, areas = self.compute_center_from_mask(gt_masks, target_class)

        min_pixels = self.min_area_ratio * float(H * W)
        target_has_point = ((areas >= min_pixels) & valid_mask).float()

        cls_loss = F.binary_cross_entropy_with_logits(pred_has_point, target_has_point)

        positive_mask = target_has_point > 0.0
        if positive_mask.any():
            valid_pred = pred_points[positive_mask].squeeze(1)
            valid_gt = gt_centers[positive_mask]

            distance = torch.norm(valid_pred - valid_gt, p=2, dim=1)
            if self.normalize_by_image_size:
                diagonal = torch.sqrt(torch.tensor(H**2 + W**2, dtype=distance.dtype, device=distance.device))
                distance = distance / diagonal
            reg_loss = distance.mean()
        else:
            reg_loss = pred_points.sum() * 0.0

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total_loss, cls_loss, reg_loss, target_has_point