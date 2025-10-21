export CUDA_VISIBLE_DEVICES=0,1,2,3 

# 使用convmae的vit-s进行视频训练，使用任务切换方法，不含有空mask约束 (原生 sam decoder)
python -m torch.distributed.launch --nproc_per_node=4 \
    ./custom_train.py ./configs/convmae/convmae_video_vits.py \
    --launcher pytorch --work-dir ./out/vits_convmae --seed 24 --deterministic \
    --options model.pretrained=./out/convmae_small.pth

# 使用convmae的vit-b进行视频训练，使用任务切换方法，不含有空mask约束 (原生 sam decoder)
python -m torch.distributed.launch --nproc_per_node=4 \
    ./custom_train.py ./configs/convmae/convmae_video_vitb.py \
    --launcher pytorch --work-dir ./out/vitb_convmae --seed 24 --deterministic \
    --options model.pretrained=./out/convmae_base.pth

