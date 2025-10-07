import os
import os.path as osp
from typing import List
import random

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DrivableVideoData(CustomDataset):
    """Video dataset for drivable area segmentation with temporal clips.

    Builds training samples as clips of ``num_frames`` consecutive images
    within each sequence folder. The annotation is for the LAST frame of the
    clip, and all preceding frames are used as temporal context.

    Assumptions:
      - Frames from the same sequence are stored under the same subdirectory
        of ``img_dir``; ordering is lexicographic on file name.
      - The split file (if provided) lists relative image paths for frames.
      - structure of dir:
        <data_root>/
            images/
                train/
                seq_0001/
                    000001.jpg
                    000002.jpg
                    000003.jpg
                    ...
                seq_0002/
                    000001.jpg
                    000002.jpg
                    ...
            annotations/
                train/
                seq_0001/
                    000001.png
                    000002.png
                    000003.png
                    ...
                seq_0002/
                    000001.png
                    000002.png
                    ...

    Additional Args:
        num_frames (int): number of frames per clip (odd is recommended).
        frame_stride (int): stride between adjacent frames within a clip.
        stride_choices (list[int] | None): optional training-time stride candidates.
            If provided, a stride will be sampled from this list per sample; otherwise
            the fixed ``frame_stride`` is used. Testing always uses a fixed stride.
    """

    CLASSES = (
        "_background_",
        "drivable_area",
    )

    PALETTE = [[0, 0, 0], [0, 255, 0]]

    def __init__(self, num_frames: int = 3, frame_stride: int = 1, stride_choices=[1, 2, 3, 4], **kwargs):
        assert num_frames >= 1 and isinstance(num_frames, int)
        assert frame_stride >= 1 and isinstance(frame_stride, int)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        # Optional training-time stride jitter choices (e.g., [1, 2, 3])
        self.stride_choices = stride_choices
        super().__init__(**kwargs)

    @staticmethod
    def _extract_timestamp_from_path(path: str):
        """Extract float timestamp from filename like
        '..._1724312702.671661.jpg'. Returns float or None if not found.
        """
        base = osp.basename(path)
        stem, _ext = osp.splitext(base)
        if '_' not in stem:
            return float(stem)
        last = stem.rsplit('_', 1)[-1]
        try:
            return float(last)
        except Exception:
            return None

    @staticmethod
    def _frame_sort_key(rel_path: str):
        """Return a robust sorting key using basename without suffix.

        If the stem is numeric (e.g., '000123'), use its integer value to ensure
        natural order; otherwise, try to extract trailing digits; fallback to stem.
        """
        base = osp.basename(rel_path)
        stem, _ext = osp.splitext(base)
        if stem.isdigit():
            return (0, float(stem))
        return (1, stem)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        # First, enumerate frames using parent behavior
        base_img_infos = super().load_annotations(img_dir, img_suffix, ann_dir, seg_map_suffix, split)

        # Group by sequence (subdirectory relative to img_dir)
        seq_to_items = {}
        for info in base_img_infos:
            rel_path = info['filename']  # relative to img_dir from dir img_dir or ann_dir
            seq_key = osp.dirname(rel_path)
            seq_to_items.setdefault(seq_key, []).append(info)
        
        # Sort items within each sequence by basename without suffix (numeric-aware)
        for key in seq_to_items:
            seq_to_items[key] = sorted(
                seq_to_items[key], key=lambda x: self._frame_sort_key(x['filename'])
            )
        
        # NOTE: removed debug print of sequence keys
        # Build contiguous windows of length ``self.num_frames`` ending at each target frame
        clip_infos = []
        for seq_key, items in seq_to_items.items():
            num_items = len(items)
            # end_idx is the target frame index (label belongs to this frame)
            for end_idx in range(num_items):
                window_start = max(0, end_idx - (self.num_frames - 1))
                window_indices = list(range(window_start, end_idx + 1))
                # Require at least self.num_frames frames available in the window during training
                # In testing, keep all frames (including the first few that don't have enough history)
                if len(window_indices) < self.num_frames and not self.test_mode:
                    continue

                candidate_paths_rel: List[str] = [items[i]['filename'] for i in window_indices]
                # collect candidate annotation paths aligned with candidate_paths_rel
                candidate_ann_paths_rel: List[str] = []
                for i in window_indices:
                    ann_rel = None
                    if 'ann' in items[i] and items[i]['ann'] is not None:
                        ann_rel = items[i]['ann'].get('seg_map', None)
                    candidate_ann_paths_rel.append(ann_rel)

                last_info = items[end_idx]
                clip_info = dict(
                    filename=last_info['filename'],
                    candidate_paths=candidate_paths_rel,
                    num_frames=self.num_frames,
                    candidate_ann_paths=candidate_ann_paths_rel,
                    group_id=seq_key,
                    group_index=end_idx,
                    group_size=num_items,
                    is_seq_start=(end_idx == 0),
                )
                if ann_dir is not None and 'ann' in last_info:
                    clip_info['ann'] = last_info['ann']
                clip_infos.append(clip_info)

        return clip_infos

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        # expose clip-level fields
        results['num_frames'] = getattr(self, 'num_frames', None)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx) if not self.test_mode else None
        results = dict(img_info=img_info)
        if ann_info is not None:
            results['ann_info'] = ann_info
        # Stride-aware contiguous sampling ending at target frame
        candidate_rel = img_info.get('candidate_paths', img_info.get('frame_paths'))
        candidate_abs = [osp.join(self.img_dir, p) for p in candidate_rel]

        T = self.num_frames
        end = len(candidate_abs) - 1
        if T <= 1:
            indices = [end]
        else:
            # Always select the last T contiguous frames ending at the target
            start = end - (T - 1)
            indices = list(range(start, end + 1))

        frame_paths_abs: List[str] = [candidate_abs[i] for i in indices]
        # enrich with paths
        results['frame_paths'] = frame_paths_abs
        # prepare corresponding seg paths for all selected frames (aux supervision)
        cand_ann_rel = img_info.get('candidate_ann_paths', None)
        if cand_ann_rel is not None:
            seg_paths_abs: List[str] = []
            for i in indices:
                rel = cand_ann_rel[i]
                seg_paths_abs.append(None if rel is None else osp.join(self.ann_dir, rel))
            results['seg_paths'] = seg_paths_abs
        # attach timestamps parsed from filenames
        frame_timestamps = [self._extract_timestamp_from_path(p) for p in frame_paths_abs]
        results['frame_timestamps'] = frame_timestamps
        results['target_timestamp'] = frame_timestamps[-1] if len(frame_timestamps) > 0 else None
        # filename is the target (last) frame for which label is provided
        results['filename'] = osp.join(self.img_dir, img_info['filename'])
        if self.ann_dir is not None and ann_info is not None:
            results['seg_filename'] = osp.join(self.ann_dir, ann_info['seg_map'])
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        # For streaming inference, load single image (target frame) and attach grouping metas
        results['filename'] = osp.join(self.img_dir, img_info['filename'])
        results['ori_filename'] = osp.basename(results['filename'])
        # Grouping metas for streaming reset
        results['group_id'] = img_info.get('group_id', osp.dirname(img_info['filename']))
        results['group_index'] = img_info.get('group_index', None)
        results['group_size'] = img_info.get('group_size', None)
        results['is_seq_start'] = bool(img_info.get('is_seq_start', False))
        results['ann'] = img_info.get('ann', None)
        self.pre_pipeline(results)
        return self.pipeline(results)


