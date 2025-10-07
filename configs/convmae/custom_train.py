_base_ = [
    '../_base_/models/upernet_video.py',
    '../_base_/datasets/drivable_video.py',
    '../_base_/default_runtime.py',
]

crop_size = (512, 512)
HISTORY_LENGTH = 4
total_epochs = 40
lr_config = dict(
    warmup_ratio=1e-2,
    warmup_iters=500,
    warmlen_ratio=0.05
)
val_epoch = 1
evaluation = dict(interval=374)
# Batch size per GPU; note each sample is a clip of num_frames
data = dict(samples_per_gpu=8, workers_per_gpu=2)

model = dict(
    history_length=HISTORY_LENGTH,
    backbone=dict(
        type='VideoConvMAE',
        img_size=[512, 128, 64],
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
        out_indices=[3, 5, 7, 11]
    ),
    decode_head=dict(
        type='PredictiveTemporalUPerHead',
        in_channels=[256, 384, 768],
        in_index=[0, 1, 2],
        num_classes=2,
        channels=256,
        streaming=True,
        # Optionally detach state every N steps to stabilize very long sequences (0 disables)
        detach_every=0,
        mask_ratio=4,
        history_length=HISTORY_LENGTH,
        use_topk_memory=True,
        topk_memory_size=256
    ),

    # auxiliary_head=dict(
    #     type='TemporalFCNHead',
    #     in_channels=768,
    #     num_classes=2,
    #     num_frames=1,
    #     temporal_fusion_method=TEMPORAL_FUSION_METHOD,
    # ),
    auxiliary_head=None,

    test_cfg=dict(mode='whole')
)


optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=11, layer_decay_rate=0.75)
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
