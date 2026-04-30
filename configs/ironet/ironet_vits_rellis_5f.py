_base_ = [
    '../_base_/models/upernet_video.py',
    '../_base_/datasets/rellis_video.py',
    '../_base_/default_runtime.py',
]


crop_size = (512, 512)
num_frames = 8
HISTORY_LENGTH = 5
total_epochs = 20
lr_config = dict(
    warmup_iters=150,
    warmup_ratio=1e-2,
    warmlen_ratio=0.05
)
val_epoch = 1
evaluation = dict(interval=411)
# Batch size per GPU; note each sample is a clip of num_frames
data = dict(samples_per_gpu=8, workers_per_gpu=4)

model = dict(
    history_length=HISTORY_LENGTH,
    backbone=dict(
        type='VideoConvMAE',
        img_size=[512, 128, 64],
        patch_size=[4, 2, 2],
        embed_dim=[128, 256, 384],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
    ),
    decode_head=dict(
        type='PredictiveTemporalUPerHead',
        in_channels=[128, 256, 384],
        in_index=[0, 1, 2],
        num_classes=2,
        channels=256,
        streaming=True,
        # Optionally detach state every N steps to stabilize very long sequences (0 disables)
        detach_every=0,
        mask_ratio=4,
        frame_stride=4,
        history_length=HISTORY_LENGTH,
        use_topk_memory=True,
        topk_memory_size=256,
        sam_prompt_test_image_embedding_size=(75//3, 120//3),
        sam_prompt_test_image_size=(1200//3, 1920//3)
    ),

    auxiliary_head=None,

    test_cfg=dict(mode='whole')
)


optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=11, layer_decay_rate=0.8)
)

# do not use mmdet version fp16 
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
