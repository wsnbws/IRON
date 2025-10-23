export CUDA_VISIBLE_DEVICES=0,1,2,3 

# # 使用convmae的vit-s进行视频训练，使用任务切换方法，含有空mask约束 (原生 sam decoder)
python -m torch.distributed.launch --nproc_per_node=4 \
    ./custom_train.py ./configs/convmae/convmae_video_vits.py \
    --launcher pytorch --work-dir ./out/vits_convmae --seed 24 --deterministic \
    --options model.pretrained=./out/convmae_small.pth

# # 使用convmae的vit-b进行视频训练，使用任务切换方法，含有空mask约束 (原生 sam decoder)
# python -m torch.distributed.launch --nproc_per_node=4 \
#     ./custom_train.py ./configs/convmae/convmae_video_vitb.py \
#     --launcher pytorch --work-dir ./out/vitb_convmae --seed 24 --deterministic \
#     --options model.pretrained=./out/convmae_base.pth

# 使用dinov3的vit-s进行视频训练，使用任务切换方法，含有空mask约束 (原生 sam decoder)
# python -m torch.distributed.launch --nproc_per_node=4 \
#     ./custom_train.py ./configs/convmae/dinov3_vits.py \
#     --launcher pytorch --work-dir ./out/dinov3_vits --seed 24 --deterministic \
#     --options model.pretrained=./out/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# # 使用dinov3的vit-b进行视频训练，使用任务切换方法，含有空mask约束 (原生 sam decoder)
# python -m torch.distributed.launch --nproc_per_node=4 \
#     ./custom_train.py ./configs/convmae/dinov3_vitb.py \
#     --launcher pytorch --work-dir ./out/dinov3_vitb --seed 24 --deterministic \
#     --options model.pretrained=./out/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth


python -m torch.distributed.launch --nproc_per_node=4 \
    ./custom_train.py ./configs/convmae/simple_temporal.py \
    --launcher pytorch --work-dir ./out/simple_dinov3_vits --seed 24 --deterministic \
    --options model.pretrained=./out/dinov3_vits16_pretrain_lvd1689m-08c60483.pth