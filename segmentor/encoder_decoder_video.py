import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from head.flag import get_test_task_state


@contextmanager
def amp_off():
    """Disable AMP autocast within the context (Apex if available else PyTorch)."""
    try:
        from apex import amp  # type: ignore
        with amp.disable_casts():
            yield
    except Exception:
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
            yield

@SEGMENTORS.register_module()
class EncoderDecoderVideo(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 history_length=0,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoderVideo, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.history_length = history_length

        self.init_weights(pretrained=pretrained)

        self.img_feature = None

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoderVideo, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        seg_logits, confidence, points = self._decode_head_forward_test(x, img_metas)
        seg_logits = resize(
            input=seg_logits,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits, confidence, points

    def _get_timestamp(self, img_metas, t):
        batch_timestamps = [] 

        for m in img_metas:
            # unwrap DataContainer if needed and create a shallow copy
            base = m.data if hasattr(m, 'data') else m
            idx = int(t)
            ts_list_all = base['frame_timestamps']
            timestamps = [float(ts_list_all[idx])] * (self.history_length + 1)
            for i, j in enumerate(range(t, -1, -1)):
                timestamps[self.history_length -i] = float(ts_list_all[j])
            batch_timestamps.append(timestamps)
        
        return batch_timestamps

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, t=None, current_iter=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # If time index is provided, mark sequence start for streaming heads
        batch_timestamps = self._get_timestamp(img_metas, t)
        ts_tensor = torch.tensor(batch_timestamps, dtype=torch.float32)
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        t=t,
                                                        timestamps=ts_tensor,
                                                        current_iter=current_iter)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        # If the decode head supports streaming and we are given single-frame inputs
        # per sample (B, C, H, W) at test time, just call forward as usual.
        seg_logits, confidence, points = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits, confidence, points

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_train(self, img, img_metas, gt_semantic_seg, step, current_iter):

        final_losses = dict()
        
        gtn = torch.stack([gt['gt_semantic_segs'].data for gt in img_metas], dim=0).to(img.device)
        
        # # 在 step == 0 时，批量计算所有帧的 backbone + FPN + PSP
        # if step == 0:
        #     B, T, C, H, W = img.shape
        #     # 1. 批量提取 backbone 特征: (B, T, C, H, W) -> (B*T, C, H, W)
        #     img_flat = img.view(B * T, C, H, W)
        #     img_feat_flat = self.extract_feat(img_flat)  # (B*T, C_feat, H_feat, W_feat)
        #     fpn_outs_flat = self.decode_head._fpn_forward_single(img_feat_flat)  # list((B*T, C_dec, H_feat, W_feat))
        #     if len(fpn_outs_flat) > 1:
        #         fpn_last_flat = fpn_outs_flat[-1]  # 使用最高分辨率特征
        #     else:
        #         fpn_last_flat = fpn_outs_flat[0]
            
        #     fpn_outs_flat[-1] = self.decode_head.psp_module(fpn_last_flat)  # (B*T, C_dec, H_feat, W_feat)
        #     self.img_feature = fpn_outs_flat
        
        # fpn_outs_step = [i.view(B, T, *i.shape[1:])[:, step] for i in self.img_feature]
        img_feature = self.extract_feat(img[:, step])
        gt_t = gtn[:, step].unsqueeze(1)

        # 因为特征已经过 FPN+PSP 处理，所以传入 skip_fpn_psp=True
        loss_decode_t = self._decode_head_forward_train(img_feature, img_metas, gt_t, t=step, current_iter=current_iter)
        for k, v in loss_decode_t.items():
            final_losses[k] = final_losses.get(k, 0) + v

        if self.with_auxiliary_head:
            loss_aux_t = self._auxiliary_head_forward_train(img_feature, img_metas, gt_t)
            for k, v in loss_aux_t.items():
                final_losses[k] = final_losses.get(k, 0) + v
                
        return final_losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit, confidence, points = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit, confidence, points

    def forward_test(self, imgs, img_metas, **kwargs):
        """Forward test for single channel mask output with sigmoid activation.
        
        Args:
            imgs (List[Tensor]): Input images (only single image supported)
            img_metas (List[List[dict]]): Image metadata 
            threshold (float): Threshold for binary segmentation, default 0.5
            rescale (bool): Whether to rescale back to original shape, default True
        """
        threshold = kwargs.pop('threshold', 0.5)
        rescale = kwargs.pop('rescale', True)
        
        # Input validation
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        # Only support single image inference
        assert len(imgs) == 1, "Only single image inference is supported"
        assert len(img_metas) == 1, "Only single image metadata is supported"
        
        img = imgs[0]
        img_meta = img_metas[0]
        
        # Get raw logits, confidence and points from whole image inference
        seg_logit, confidence, points = self.whole_inference(img, img_meta, rescale)
        
        # Handle image flipping if specified
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_logit = seg_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_logit = seg_logit.flip(dims=(2, ))
        
        # Apply sigmoid activation for single channel output
        seg_prob = torch.sigmoid(seg_logit)
        # Apply threshold to get binary mask (0: background, 1: drivable area)
        seg_pred = (seg_prob > threshold).long()

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred, confidence, points
        
        seg_pred = seg_pred.cpu().numpy()

        return seg_pred, confidence, points