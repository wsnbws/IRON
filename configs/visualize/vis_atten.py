# Visualization config for scripts/vis_atten.py
# You can modify paths and parameters below as needed.

vis_atten = dict(
    # Path to the .npy attention weight file
    npy_file="/home/wangshuo/otdr/out/atten_weights/20240822-xts-day-5-KAAtil_1724315343.083632.jpg_cross_attn_weight_5.npy",

    # Directory containing the original images searched by basename
    # base_image_dir="/data20t/wangshuo/IR_Drivable/OTDR/test/images/xts_5",
    base_image_dir="/data20t/wangshuo/IR_Drivable/baseline/images/test/xts_video",

    # Patch index to visualize
    patch_idx=795,

    # Number of history frames; set to None to infer from data length
    frames=6,

    # Feature grid spatial size (H, W)
    spatial_h=32,
    spatial_w=40,

    # Output root directory for generated visualizations
    output_root="/home/wangshuo/otdr/out/atten_vis",
)


