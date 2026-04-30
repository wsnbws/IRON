import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class TemporalQueue(nn.Module):
    """
    Configurable temporal queue for storing historical frames and masks.
    
    This queue maintains a sliding window of historical data with configurable size.
    It provides the same interface as the original hardcoded 2-frame queue but allows
    for future extension to arbitrary history lengths.
    
    Args:
        history_length (int): Number of historical frames to store. Default: 2
        streaming (bool): Whether to enable streaming mode. Default: False
    """
    def __init__(self, history_length: int = 2, streaming: bool = False):
        super(TemporalQueue, self).__init__()
        self.history_length = history_length
        self.streaming = streaming
        
        # Queue storage tensors - will be allocated dynamically
        self._queue_feats: Optional[torch.Tensor] = None
        self._queue_masks: Optional[torch.Tensor] = None
        
    def ensure_allocation(self, feat: torch.Tensor, mask_shape: Tuple[int, ...]) -> None:
        """
        Allocate or reallocate queue tensors to match current batch and feature shape.
        
        Args:
            feat: Sample feature tensor to determine shape (B, C, H, W)
            mask_shape: Shape of mask tensors (num_classes-1, mask_H, mask_W)
        """
        if not self.streaming:
            return
            
        B, C, H, W = feat.shape
        
        # Check if reallocation is needed for features
        if (self._queue_feats is None or 
            self._queue_feats.shape[1] != B or 
            self._queue_feats.shape[2:] != (C, H, W)):
            self._queue_feats = torch.zeros(
                (self.history_length, B, C, H, W), 
                dtype=feat.dtype, 
                device=feat.device
            )
            
        # Check if reallocation is needed for masks
        if (self._queue_masks is None or 
            self._queue_masks.shape[1] != B or 
            self._queue_masks.shape[2:] != mask_shape):
            self._queue_masks = torch.zeros(
                (self.history_length, B, *mask_shape), 
                dtype=feat.dtype, 
                device=feat.device
            )
    
    def reset_state(self,  all_batch: bool = False) -> None:
        """
        Reset queue memory for selected slots or all.
        
        Args:
            mask: Bool tensor of shape (B,) for batch positions to reset. 
                  If None and all_batch False, do nothing.
            all_batch: If True, reset all batch positions.
        """
        if not self.streaming or self._queue_feats is None:
            return
            
        if all_batch:
            self._queue_feats.zero_()
            self._queue_masks.zero_()
            return
    
    def detach_state(self) -> None:
        """Detach queue tensors from computation graph."""
        if not self.streaming:
            return
        if self._queue_feats is not None:
            self._queue_feats = self._queue_feats.detach()
        if self._queue_masks is not None:
            self._queue_masks = self._queue_masks.detach()
    
    def push(self, feat: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Push new feature and mask to the queue.
        
        Args:
            feat: Feature tensor of shape (B, C, H, W)
            mask: Mask tensor of shape (B, num_classes-1, mask_H, mask_W)
        """
        if not self.streaming or self._queue_feats is None:
            return
            
        with torch.no_grad():
            # Shift queue: move all frames back by one position
            self._queue_feats = torch.roll(self._queue_feats, shifts=-1, dims=0)
            self._queue_feats[-1] = feat.detach()

            # Update current size
            self._queue_masks = torch.roll(self._queue_masks, shifts=-1, dims=0)
            self._queue_masks[-1] = mask.detach()
    
    def get_frame(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get feature and mask at specific historical index.
        
        Args:
            index: Historical index (0 = oldest, history_length-1 = newest)
            
        Returns:
            Tuple of (feature_tensor, mask_tensor)
            
        Note:
            For backward compatibility with the original 2-frame implementation:
            - index 0 returns t-2 (oldest available)
            - index 1 returns t-1 (second newest)
        """ 
        feat = self._queue_feats[index].clone()
        mask = self._queue_masks[index].clone()
        return feat, mask
    
    def get_history_frames(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all historical frames for backward compatibility.
        
        Returns:
            Tuple of (all_features, all_masks) where each has shape 
            (history_length, B, ...)
        """
        if not self.streaming or self._queue_feats is None:
            return None, None
        return self._queue_feats.clone(), self._queue_masks.clone() 

