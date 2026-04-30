# dataset settings for temporal video segmentation

dataset_type = 'DrivableVideoData'
# data_root = '/data20t/wangshuo/IR_Drivable/baseline'
data_root = '/data20t/wangshuo/orfd_iofs'
# data_root = '/data20t/wangshuo/IR_Drivable/OTDR/test/images'

# number of frames per clip and temporal stride
num_frames = 8
frame_stride = 4

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadMultiImageFromFile'),
    dict(type='LoadAnnotationsVideo', reduce_zero_label=False),
    # Use a fixed or multi-scale policy; ResizeMulti supports list of scales
    dict(type='ResizeMulti', img_scale=(2048, 512), ratio_range=(0.5, 2)),
    dict(type='RandomCropMulti', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipMulti', prob=0.5, direction='horizontal'),
    # 时序抖动增强 - 对前面帧添加轻微扰动
    dict(type='PhotoMetricDistortionMulti'),
    # dict(type='TemporalConsistencyLoss', prob=0.6, jitter_std=0.03, frame_dropout_prob=0.15),
    # 激进的最后一帧遮挡 - 迫使依赖memory
    # dict(type='RandomOcclusionLastFrame', prob=0.8, aggressive_mode=True, fill_value=0),
    dict(type='NormalizeMulti', **img_norm_cfg),
    dict(type='PadMulti', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundleVideo'),
    dict(type='CollectVideo', keys=['img', 'gt_semantic_seg']),
]

# test_pipeline = [
#     dict(type='LoadMultiImageFromFile'),
#     dict(
#         type='MultiScaleFlipAugVideo',
#         img_scale=(2048, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='ResizeMulti', keep_ratio=True),
#             dict(type='RandomFlipMulti'),
#             dict(type='NormalizeMulti', **img_norm_cfg),
#             # Map to tensor like ImageToTensor in static config
#             dict(type='ImageToTensorMulti', keys=['imgs']),
#             # Collect with 'img' key compatibility
#             dict(type='CollectVideo', keys=['img']),
#         ],
#     ),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 720),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='CollectVideo', keys=['img']),
#         ],
#     ),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048,360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(512,640), pad_val=0, seg_pad_val=0),  # 在 Normalize 前
            dict(type='ImageToTensor', keys=['img']),
            dict(type='CollectVideo', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/anns',
        img_suffix='.png',
        seg_map_suffix='.png',
        num_frames=num_frames,
        frame_stride=frame_stride,
        pipeline=train_pipeline,
        stride_choices=[1,2],
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/anns',
        img_suffix='.png',
        seg_map_suffix='.png',
        num_frames=num_frames,
        frame_stride=frame_stride,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/anns',
        img_suffix='.png',
        seg_map_suffix='.png',
        num_frames=num_frames,
        frame_stride=frame_stride,
        pipeline=test_pipeline,
    ),
)


