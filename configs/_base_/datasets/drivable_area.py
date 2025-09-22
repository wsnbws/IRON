# dataset settings
dataset_type = 'DrivableData'
data_root = '/data20t/wangshuo/IR_Drivable/baseline'


img_norm_cfg = dict(
     mean=[108.375,108.375,108.375], std=[50.7195,50.7195,50.7195], to_rgb=True)

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Contrast_Aug', power=0.7),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512), 
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Contrast_Aug', power=0.7),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

IMG_DIR =  'images/training/xts'
ANN_DIR =    'anns/training/xts'
TEST_IMG_DIR = 'images/test/xts'
TEST_ANN_DIR =   'anns/test/xts'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir= IMG_DIR,
        ann_dir= ANN_DIR,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir= TEST_IMG_DIR,
        ann_dir= TEST_ANN_DIR,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir= TEST_IMG_DIR,
        ann_dir= TEST_ANN_DIR,
        pipeline=test_pipeline))
