# Visualization config for scripts/vis_atten.py
# You can modify paths and parameters below as needed.

vis_atten = dict(
    # Path to the .npy attention weight file
    npy_file="/home/wangshuo/otdr/out/atten_weights/20240822-xts-day-5-KAAtil_1724315190.082782.jpg_cross_attn_weight_3.npy",

    # Directory containing the original images searched by basename
    # base_image_dir="/data20t/wangshuo/IR_Drivable/OTDR/test/images/xts_5",
    base_image_dir="/data20t/wangshuo/IR_Drivable/baseline/images/test/xts_video",

    # Patch index to visualize
    patch_idx=960,

    # Number of history frames; set to None to infer from data length
    frames=3,

    # Feature grid spatial size (H, W)
    spatial_h=32,
    spatial_w=40,

    # Output root directory for generated visualizations
    output_root="/home/wangshuo/otdr/out/atten_vis",
)


