import os
import os.path as osp
from typing import List, Tuple

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadMultiImageFromFile(object):
    """Load multiple consecutive RGB images for a video clip.

    Expects ``results`` to contain:
      - 'frame_paths': List[str] absolute paths of length T ordered by time
      - 'filename':    str path of the LAST frame (target frame)

    It will set:
      - 'imgs': List[np.ndarray] list of HxWx3 uint8
      - 'img':  np.ndarray of the LAST frame (for compatibility with single-frame ops)
      - 'ori_shape', 'img_shape', 'pad_shape'
      - 'img_fields' to include 'imgs' and 'img'
    """

    def __init__(self, to_float32: bool = False, color_type: str = 'color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results: dict) -> dict:
        frame_paths: List[str] = results.get('frame_paths', None)
        assert isinstance(frame_paths, list) and len(frame_paths) > 0, \
            'LoadMultiImageFromFile requires results["frame_paths"] as a non-empty list.'

        imgs = []
        for p in frame_paths:
            img = mmcv.imread(p, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        # Use the LAST frame as the compatibility 'img' (target frame)
        target_index = len(imgs) - 1
        target_img = imgs[target_index]
        results['ori_filename'] =  os.path.basename(frame_paths[target_index])
        results['imgs'] = imgs
        results['img'] = target_img
        results['ori_shape'] = target_img.shape
        results['img_shape'] = target_img.shape
        results['pad_shape'] = target_img.shape
        results.setdefault('img_fields', [])
        if 'img' not in results['img_fields']:
            results['img_fields'].append('img')
        if 'imgs' not in results['img_fields']:
            results['img_fields'].append('imgs')

        # Fallback: if timestamps not provided by dataset, parse here and attach
        if 'frame_timestamps' not in results:
            def _extract_ts(path: str):
                base = os.path.basename(path)
                stem, _ext = os.path.splitext(base)
                if '_' not in stem:
                    return None
                last = stem.rsplit('_', 1)[-1]
                try:
                    return float(last)
                except Exception:
                    return None
            frame_ts = [_extract_ts(p) for p in frame_paths]
            results['frame_timestamps'] = frame_ts
            results['target_timestamp'] = frame_ts[-1] if len(frame_ts) > 0 else None
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(to_float32={self.to_float32}, color_type={self.color_type})'


@PIPELINES.register_module()
class ResizeMulti(object):
    """Resize all frames in 'imgs' consistently; also resizes 'img' and seg.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing.
        ratio_range (tuple[float] | None): When set, randomly sample a ratio
            r in [min_ratio, max_ratio] and multiply the base scale with r.
            When provided, ``img_scale`` must be None or a single scale.
    """

    def __init__(self, img_scale=None, keep_ratio=True, ratio_range=None):
        if img_scale is None:
            self.img_scale = None
        else:
            self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)
        self.keep_ratio = keep_ratio
        self.ratio_range = ratio_range
        if self.ratio_range is not None:
            # Align with mmseg Resize: either use current image size as base
            # or a single provided base scale when ratio_range is enabled.
            assert self.img_scale is None or len(self.img_scale) == 1

    @staticmethod
    def _random_sample_ratio(img_scale: tuple, ratio_range: tuple):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = (int(img_scale[0] * ratio), int(img_scale[1] * ratio))
        return scale

    def _resize_keep_ratio_get_target(self, img: np.ndarray, scale: Tuple[int, int]):
        # Follow mmseg Resize: imrescale returns a single scale factor
        resized, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        new_h, new_w = resized.shape[:2]
        h, w = img.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        return (new_w, new_h), w_scale, h_scale

    def __call__(self, results: dict) -> dict:
        assert 'imgs' in results, 'ResizeMulti requires results["imgs"]'
        imgs: List[np.ndarray] = results['imgs']

        # determine scale: prefer externally provided scale (e.g., TTA), else use own policy
        if 'scale' in results and results['scale'] is not None:
            target_scale = results['scale']
        else:
            if self.ratio_range is not None:
                # ratio-based sampling: base on current image size or a single provided scale
                if self.img_scale is None:
                    h, w = imgs[-1].shape[:2]
                    base_scale = (w, h)
                else:
                    base_scale = self.img_scale[0]
                target_scale = self._random_sample_ratio(base_scale, self.ratio_range)
            else:
                if self.img_scale is None:
                    h, w = imgs[0].shape[:2]
                    target_scale = (w, h)
                else:
                    # single-scale or list of scales: pick one randomly for augmentation
                    if len(self.img_scale) == 1:
                        target_scale = self.img_scale[0]
                    else:
                        target_scale = self.img_scale[np.random.randint(len(self.img_scale))]

        # compute target output size and scale factors based on LAST frame
        target = len(imgs) - 1
        target_img = imgs[target]
        if self.keep_ratio:
            (new_w, new_h), w_scale, h_scale = self._resize_keep_ratio_get_target(target_img, target_scale)
        else:
            resized_target, w_scale, h_scale = mmcv.imresize(target_img, target_scale, return_scale=True)
            new_h, new_w = resized_target.shape[:2]

        # resize all frames to identical size
        resized_imgs = []
        resized_gts = [] if 'gt_semantic_segs' in results else None
        for img in imgs:
            if self.keep_ratio:
                out = mmcv.imresize(img, (new_w, new_h))
            else:
                out = mmcv.imresize(img, target_scale)
            resized_imgs.append(out)
        # resize multi-frame GTs consistently
        if resized_gts is not None:
            for gt in results['gt_semantic_segs']:
                if gt is None:
                    resized_gts.append(None)
                else:
                    if self.keep_ratio:
                        gto = mmcv.imresize(gt, (new_w, new_h), interpolation='nearest')
                    else:
                        gto = mmcv.imresize(gt, target_scale, interpolation='nearest')
                    resized_gts.append(gto)
            results['gt_semantic_segs'] = resized_gts

        results['imgs'] = resized_imgs
        # update compatibility keys using LAST frame
        results['img'] = resized_imgs[target]
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results['img_shape'] = results['img'].shape
        results['pad_shape'] = results['img'].shape
        results['scale'] = target_scale
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

        # resize seg if present
        for key in results.get('seg_fields', []):
            if key == 'gt_semantic_seg':
                if self.keep_ratio:
                    results[key] = mmcv.imresize(results[key], (new_w, new_h), interpolation='nearest')
                else:
                    results[key] = mmcv.imresize(results[key], target_scale, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(img_scale={self.img_scale}, keep_ratio={self.keep_ratio}, ratio_range={self.ratio_range})'


@PIPELINES.register_module()
class RandomFlipMulti(object):
    """Flip all frames in 'imgs' consistently; also flips 'img' and seg.

    Args:
        prob (float): flipping probability.
        direction (str): 'horizontal' or 'vertical'.
    """

    def __init__(self, prob: float = 0.5, direction: str = 'horizontal'):
        self.prob = prob
        assert 0.0 <= prob <= 1.0
        assert direction in ['horizontal', 'vertical']
        self.direction = direction

    def __call__(self, results: dict) -> dict:
        # Respect pre-set flip decision if exists to keep deterministic behavior across pipelines
        if 'flip' in results:
            do_flip = results['flip']
        else:
            do_flip = np.random.rand() < self.prob
        results['flip'] = do_flip
        if 'flip_direction' in results:
            direction = results['flip_direction']
        else:
            direction = self.direction
        results['flip_direction'] = direction

        if not do_flip:
            return results

        if 'imgs' in results:
            results['imgs'] = [mmcv.imflip(img, direction=direction) for img in results['imgs']]
            # sync compatibility key to LAST frame
            results['img'] = results['imgs'][-1]
        # flip multi-frame GTs if present; keep single GT synced without double processing
        if 'gt_semantic_segs' in results:
            flipped = []
            for gt in results['gt_semantic_segs']:
                if gt is None:
                    flipped.append(None)
                else:
                    flipped.append(mmcv.imflip(gt, direction=direction))
            results['gt_semantic_segs'] = flipped

        for key in results.get('seg_fields', []):
            results[key] = mmcv.imflip(results[key], direction=direction).copy()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob}, direction={self.direction})'


@PIPELINES.register_module()
class NormalizeMulti(object):
    """Normalize all frames in 'imgs' consistently as in single-frame Normalize.

    Args:
        mean (sequence): per-channel means
        std (sequence): per-channel stds
        to_rgb (bool): if convert BGR to RGB
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results: dict) -> dict:
        assert 'imgs' in results, 'NormalizeMulti requires results["imgs"]'
        normed = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['imgs']]
        results['imgs'] = normed
        # keep compatibility LAST frame as target
        results['img'] = normed[-1]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'


@PIPELINES.register_module()
class PadMulti(object):
    """Pad all frames in 'imgs' and seg to a fixed size or divisor consistently."""

    def __init__(self, size=None, size_divisor=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img: np.ndarray):
        if self.size is not None:
            return mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
        else:
            return mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)

    def __call__(self, results: dict) -> dict:
        assert 'imgs' in results, 'PadMulti requires results["imgs"]'
        # Compute pad target based on LAST frame to ensure identical sizes
        target = len(results['imgs']) - 1
        if self.size is not None:
            target_shape = self.size
        else:
            h, w = results['imgs'][target].shape[:2]
            # find smallest (>=) multiple of size_divisor
            pad_h = int(np.ceil(h / self.size_divisor) * self.size_divisor)
            pad_w = int(np.ceil(w / self.size_divisor) * self.size_divisor)
            target_shape = (pad_h, pad_w)

        imgs = [mmcv.impad(img, shape=target_shape, pad_val=self.pad_val) for img in results['imgs']]
        # pad multi-frame GTs if present
        if 'gt_semantic_segs' in results:
            gts = []
            for gt in results['gt_semantic_segs']:
                if gt is None:
                    gts.append(None)
                else:
                    gts.append(mmcv.impad(gt, shape=target_shape, pad_val=self.seg_pad_val))
            results['gt_semantic_segs'] = gts
        results['imgs'] = imgs
        results['img'] = imgs[target]
        results['pad_shape'] = results['img'].shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key], shape=results['pad_shape'][:2], pad_val=self.seg_pad_val)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(size={self.size}, size_divisor={self.size_divisor}, pad_val={self.pad_val})'


@PIPELINES.register_module()
class RandomCropMulti(object):
    """Randomly crop all frames and seg consistently with a shared bbox.

    Args:
        crop_size (tuple): (h, w) after cropping
        cat_max_ratio (float): maximum ratio that a single category could occupy
        ignore_index (int): ignore label in seg map
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        return img[crop_y1:crop_y2, crop_x1:crop_x2, ...]

    def __call__(self, results: dict) -> dict:
        assert 'imgs' in results, 'RandomCropMulti requires results["imgs"]'
        imgs = results['imgs']
        # Use LAST frame as reference for crop window sampling
        ref_img = imgs[-1]

        crop_bbox = self.get_crop_bbox(ref_img)
        if self.cat_max_ratio < 1. and 'gt_semantic_seg' in results:
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(ref_img)

        # crop all frames
        cropped_imgs = [self.crop(img, crop_bbox) for img in imgs]
        results['imgs'] = cropped_imgs
        results['img'] = cropped_imgs[-1]
        results['img_shape'] = results['img'].shape

        # crop seg fields consistently (single GT)
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        # crop multi-frame GTs consistently
        if 'gt_semantic_segs' in results:
            cropped_gts = []
            for gt in results['gt_semantic_segs']:
                if gt is None:
                    cropped_gts.append(None)
                else:
                    cropped_gts.append(self.crop(gt, crop_bbox))
            results['gt_semantic_segs'] = cropped_gts

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class ImageToTensorMulti(object):
    """Convert images to torch.Tensor consistently for video.

    For key 'imgs': list of (H, W, C) arrays -> tensor of shape (T, C, H, W).
    For other image-like keys: behavior matches ImageToTensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results: dict) -> dict:
        for key in self.keys:
            if key not in results:
                continue
            val = results[key]
            if key == 'imgs' and isinstance(val, list):
                tensors = []
                for img in val:
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, -1)
                    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
                    tensors.append(tensor)
                stacked = torch.stack(tensors, dim=0)
                results[key] = stacked
                # For mmseg compatibility, also expose as 'img'
                results['img'] = stacked
            else:
                img = val
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results[key] = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@PIPELINES.register_module()
class LoadAnnotationsVideo(object):
    """Load annotation for video segmentation (label of the LAST frame).

    Behavior mirrors mmseg's LoadAnnotations but targets the last frame label
    only, consistent with DrivableVideoData where the last frame is the target.

    Args:
        reduce_zero_label (bool): Whether to reduce label values by 1. Default: False.
        file_client_args (dict): mmcv FileClient args. Default: dict(backend='disk').
        imdecode_backend (str): Backend for imdecode. Default: 'pillow'.
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results: dict) -> dict:
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # Multi-frame GT loading if seg_paths list is provided
        if 'seg_paths' in results and isinstance(results['seg_paths'], list):
            gts = []
            for seg_path in results['seg_paths']:
                if seg_path is None:
                    gts.append(None)
                    continue
                img_bytes = self.file_client.get(seg_path)
                gt = mmcv.imfrombytes(
                    img_bytes, flag='unchanged', backend=self.imdecode_backend).squeeze().astype(np.uint8)
                if results.get('label_map', None) is not None:
                    for old_id, new_id in results['label_map'].items():
                        gt[gt == old_id] = new_id
                if self.reduce_zero_label:
                    gt[gt == 0] = 255
                    gt = gt - 1
                    gt[gt == 254] = 255
                gts.append(gt)
            results['gt_semantic_segs'] = gts  # list aligned with frames
            # keep last frame single GT for legacy keys
            if len(gts) > 0 and gts[-1] is not None:
                results['gt_semantic_seg'] = gts[-1]
        else:
            # Prefer explicitly provided absolute path
            if 'seg_filename' in results:
                seg_path = results['seg_filename']
            else:
                # Fall back to ann_info + seg_prefix behavior
                if results.get('seg_prefix', None) is not None:
                    seg_path = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
                else:
                    seg_path = results['ann_info']['seg_map']

            img_bytes = self.file_client.get(seg_path)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend).squeeze().astype(np.uint8)

            if results.get('label_map', None) is not None:
                for old_id, new_id in results['label_map'].items():
                    gt_semantic_seg[gt_semantic_seg == old_id] = new_id

            if self.reduce_zero_label:
                gt_semantic_seg[gt_semantic_seg == 0] = 255
                gt_semantic_seg = gt_semantic_seg - 1
                gt_semantic_seg[gt_semantic_seg == 254] = 255

            results['gt_semantic_seg'] = gt_semantic_seg

        results.setdefault('seg_fields', [])
        if 'gt_semantic_seg' in results and 'gt_semantic_seg' not in results['seg_fields']:
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(reduce_zero_label={self.reduce_zero_label}, '
                f"imdecode_backend='{self.imdecode_backend}')")

@PIPELINES.register_module()
class DefaultFormatBundleVideo(object):
    """Format bundle for video clips.

    - imgs: (T, H, W, C) -> (T, C, H, W), stack to torch.Tensor and DC(stack=True)
    - gt_semantic_seg: keep single-frame label for last frame
    """

    def __call__(self, results: dict) -> dict:
        assert 'imgs' in results, 'DefaultFormatBundleVideo requires results["imgs"]'

        imgs = results['imgs']
        processed = []
        for img in imgs:
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            processed.append(img.transpose(2, 0, 1))
        imgs_tensor = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in processed], dim=0)
        # For temporal models, we keep (T, C, H, W) per sample; mmseg expects 'img' key.
        results['img'] = DC(imgs_tensor, stack=True)

        if 'gt_semantic_seg' in results:
            seg = results['gt_semantic_seg'].astype(np.int64)
            seg = torch.from_numpy(seg[None, ...])
            results['gt_semantic_seg'] = DC(seg, stack=True)

        # Package multi-frame segs if present
        if 'gt_semantic_segs' in results:
            seg_list = results['gt_semantic_segs']
            # determine target H,W
            hw = None
            for s in seg_list:
                if s is not None:
                    hw = s.shape[:2]
                    break
            if hw is None:
                # fallback to img_shape
                h, w = results.get('img_shape', imgs_tensor.shape[-2:])
                hw = (h, w)
            filled = []
            for s in seg_list:
                if s is None:
                    filled.append(np.full(hw, 255, dtype=np.uint8))
                else:
                    filled.append(s.astype(np.uint8))
            segs_np = np.stack(filled, axis=0).astype(np.int64)  # (T, H, W)
            segs_t = torch.from_numpy(segs_np)
            results['gt_semantic_segs'] = DC(segs_t, stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class CollectVideo(object):
    """Collect keys for video segmentation.

    This mirrors mmseg's Collect but passes a 5D 'img' tensor (B, T, C, H, W)
    downstream after collation. Meta information is kept for the center frame
    and clip-level fields (e.g., frame_paths, num_frames).
    """

    def __init__(self, keys, meta_keys=(
        'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'num_frames',
        'frame_paths', 'frame_timestamps', 'target_timestamp', 'gt_semantic_segs',
        'group_id', 'group_index', 'group_size', 'is_seq_start','ann'
    )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results: dict) -> dict:
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class MultiScaleFlipAugVideo(object):
    """Test-time augmentation with multiple scales and flipping for video clips.

    Interface mirrors mmseg's MultiScaleFlipAug, but expects temporal transforms
    (e.g., ResizeMulti, RandomFlipMulti, NormalizeMulti, PadMulti, etc.).

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Image scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        from mmseg.datasets.pipelines.compose import Compose  # defer import
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios, list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(img_ratios, float):
            assert len(img_scale) == 2
            self.img_scale = [(int(img_scale[0] * ratio), int(img_scale[1] * ratio)) for ratio in img_ratios]
        else:
            self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)

    def __call__(self, results: dict):
        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            # For video, use the last frame to derive original size for ratio mode
            assert 'imgs' in results and len(results['imgs']) > 0
            h, w = results['imgs'][-1].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio)) for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str

@PIPELINES.register_module()
class RandomGammaMulti(object):
    def __init__(self, gamma_range=(0.9, 1.1), prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        gamma = np.random.uniform(*self.gamma_range)
        imgs = results['imgs']
        out = []
        for img in imgs:
            imgf = img.astype(np.float32) / 255.0
            imgf = np.clip(np.power(imgf, gamma), 0, 1) * 255.0
            out.append(imgf.astype(img.dtype))
        results['imgs'] = out
        results['img'] = out[-1]
        return results


@PIPELINES.register_module()
class RandomOcclusionLastFrame(object):
    """随机遮挡最后一帧，迫使模型依赖前序帧的memory信息。
    
    Args:
        prob (float): 应用遮挡的概率
        occlusion_ratio (tuple): 遮挡区域比例范围 (min_ratio, max_ratio)
        num_blocks (tuple): 遮挡块数量范围 (min_blocks, max_blocks)
        fill_value (int): 遮挡填充值，默认为0（黑色）
        aggressive_mode (bool): 激进模式，使用更强的遮挡策略
    """
    
    def __init__(self, prob=0.5, occlusion_ratio=(0.1, 0.3), num_blocks=(1, 3), fill_value=0, aggressive_mode=False):
        self.prob = prob
        self.occlusion_ratio = occlusion_ratio
        self.num_blocks = num_blocks
        self.fill_value = fill_value
        self.aggressive_mode = aggressive_mode
        
    def __call__(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
            
        assert 'imgs' in results, 'RandomOcclusionLastFrame requires results["imgs"]'
        imgs = results['imgs']
        
        # 只对最后一帧进行遮挡
        last_img = imgs[-1].copy()
        h, w = last_img.shape[:2]
        
        if self.aggressive_mode:
            # 激进模式：随机选择一种强遮挡策略
            mode = np.random.choice(['blocks', 'strips', 'grid', 'center'])
            
            if mode == 'blocks':
                # 大块遮挡
                n_blocks = np.random.randint(2, 6)
                for _ in range(n_blocks):
                    ratio = np.random.uniform(0.15, 0.4)
                    block_h = int(h * np.sqrt(ratio))
                    block_w = int(w * np.sqrt(ratio))
                    top = np.random.randint(0, h - block_h + 1)
                    left = np.random.randint(0, w - block_w + 1)
                    last_img[top:top+block_h, left:left+block_w] = self.fill_value
                    
            elif mode == 'strips':
                # 条纹遮挡
                n_strips = np.random.randint(3, 8)
                for _ in range(n_strips):
                    if np.random.rand() > 0.5:  # 水平条纹
                        strip_h = np.random.randint(h//20, h//8)
                        top = np.random.randint(0, h - strip_h + 1)
                        last_img[top:top+strip_h, :] = self.fill_value
                    else:  # 垂直条纹
                        strip_w = np.random.randint(w//20, w//8)
                        left = np.random.randint(0, w - strip_w + 1)
                        last_img[:, left:left+strip_w] = self.fill_value
                        
            elif mode == 'grid':
                # 网格遮挡
                grid_size = np.random.randint(30, 80)
                for i in range(0, h, grid_size*2):
                    for j in range(0, w, grid_size*2):
                        if np.random.rand() > 0.5:
                            end_i = min(i + grid_size, h)
                            end_j = min(j + grid_size, w)
                            last_img[i:end_i, j:end_j] = self.fill_value
                            
            elif mode == 'center':
                # 中心区域大面积遮挡（最关键的驾驶区域）
                center_ratio = np.random.uniform(0.3, 0.6)
                ch, cw = int(h * center_ratio), int(w * center_ratio)
                top = (h - ch) // 2
                left = (w - cw) // 2
                last_img[top:top+ch, left:left+cw] = self.fill_value
        else:
            # 原有的基础遮挡策略
            n_blocks = np.random.randint(self.num_blocks[0], self.num_blocks[1] + 1)
            
            for _ in range(n_blocks):
                ratio = np.random.uniform(*self.occlusion_ratio)
                block_h = int(h * np.sqrt(ratio))
                block_w = int(w * np.sqrt(ratio))
                top = np.random.randint(0, h - block_h + 1)
                left = np.random.randint(0, w - block_w + 1)
                last_img[top:top+block_h, left:left+block_w] = self.fill_value
        
        # 更新结果
        results['imgs'][-1] = last_img
        results['img'] = last_img
        
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob}, occlusion_ratio={self.occlusion_ratio}, num_blocks={self.num_blocks}, fill_value={self.fill_value}, aggressive_mode={self.aggressive_mode})'


@PIPELINES.register_module()
class TemporalConsistencyLoss(object):
    """在数据层面增加时序一致性约束，迫使模型关注帧间差异。
    
    通过对连续帧做不同程度的扰动，增强时序依赖。
    """
    
    def __init__(self, prob=0.5, jitter_std=0.02, frame_dropout_prob=0.1):
        self.prob = prob
        self.jitter_std = jitter_std
        self.frame_dropout_prob = frame_dropout_prob
        
    def __call__(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
            
        assert 'imgs' in results, 'TemporalConsistencyLoss requires results["imgs"]'
        imgs = results['imgs']
        
        # 对前面的帧添加轻微jitter，模拟相机抖动
        for i in range(len(imgs) - 1):  # 保持最后一帧不变
            img = imgs[i].astype(np.float32)
            
            # 添加高斯噪声
            noise = np.random.normal(0, self.jitter_std * 255, img.shape)
            img_jittered = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # 随机dropout某些帧（用前一帧替代）
            if i > 0 and np.random.rand() < self.frame_dropout_prob:
                imgs[i] = imgs[i-1].copy()
            else:
                imgs[i] = img_jittered
        
        results['imgs'] = imgs
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob}, jitter_std={self.jitter_std}, frame_dropout_prob={self.frame_dropout_prob})'