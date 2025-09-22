#### 数据pipeline
---
```
# img_infos
[
    img_info={
    filename: 相对目前目录的图片路径+后缀，
    ann: {
        seg_map: 相对目前目录的标注图片名称+后缀
    }
},...
]
# clip_infos 视频下的img_infos
[
    clip_info={
        filename: 连续帧的最后一帧图片路径+后缀，
        frame_paths: [连续帧的每一帧filename]，
        num_frames: 连续帧帧数,
        ann: {seg_map: 最后一帧相对标注图片名称+后缀}
    },...
]

results={

   img_info: img_infos[idx]，
   ann_info: img_infos[idx]['ann'],
   frame_paths: 当前帧对应的一段连续帧的图片完整地址列表，
   filename: img_infos[idx] 全地址，
   seg_filename: img_infos[idx]['ann']['seg_map'] 全地址，
   img_prefix: 图片的目录地址 /path/to/dataset/images/training,
   seg_prefix: 标注的目录地址 /path/to/dataset/ann/training,
   seg_fields: [],
   label_map: {}
   "一些数据增强过程中添加的键值对"，
   num_frames: 连续帧数

   # LoadMultiImageFromFile
   imgs: [连续帧的图像数据]，
   img: 最后一帧的图像数据，
   ori_shape = img_shape = pad_shape: img shape,
   img_fields:['img', 'imgs']

   # LoadAnnotationsVideo
   gt_semantic_seg: 标注图像原始数据
   更新seg_fields -> ['gt_semantic_seg']

   # ResizeMulti
   更新 img, imgs, img_shape, pad_shape, gt_semantic_seg
   scale: 图像的resize的(w, h),
   scale_factor: scale的扩缩因子，
   keep_ratio: True(default)

   # RandomCropMulti
   更新 img, imgs, img_shape, gt_semantic_seg

   # RandomFlipMulti
   更新 imgs, img, gt_semantic_seg
   flip: do if random prob < 0.5
   flip_direction: horizontal

   # NormalizeMulti
   更新 img, imgs
   img_norm_cfg: {mean, std, to_rgb}

   # PadMulti
   更新 img, imgs, pad_shape
   pad_fixed_size
}
```


#### 数据最后输出，模型输入的格式
---
```
# CollectVideo 
CollectVideo = {
    img: results['img'],
    gt_semantic_seg: result['gt_semantic_seg'],
    img_metas: {
        'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'num_frames','frame_paths'
    }
}
```