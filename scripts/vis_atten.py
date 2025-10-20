import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from mmcv.utils import Config
import argparse

def _load_vis_cfg(cfg_path=None):
    """从配置文件读取可视化参数。
    
    优先使用命令行传入的 --config 路径；若未提供，则读取环境变量
    VIS_ATTEN_CFG；仍未提供时回退到 '/home/wangshuo/otdr/configs/vis_atten.py'。
    
    期望的配置结构（两种之一）：
    1) 直接在根上定义字段：npy_file, base_image_dir, patch_idx, frames,
       spatial_h, spatial_w, output_root
    2) 在 cfg.vis_atten 下面定义同名字段
    """
    default_cfg_path = cfg_path or os.environ.get('VIS_ATTEN_CFG', '/home/wangshuo/otdr/configs/vis_atten.py')
    cfg_values = {}
    if os.path.isfile(default_cfg_path):
        try:
            cfg_obj = Config.fromfile(default_cfg_path)
            if isinstance(cfg_obj, dict) and 'vis_atten' in cfg_obj:
                cfg_values = cfg_obj['vis_atten']
            elif hasattr(cfg_obj, 'get'):
                cfg_values = cfg_obj.get('vis_atten', cfg_obj)
            else:
                cfg_values = cfg_obj
            print(f"Loaded config from: {default_cfg_path}")
        except Exception as e:
            print(f"Failed to load config '{default_cfg_path}': {e}. Using defaults.")
    else:
        print(f"Config file not found: {default_cfg_path}. Using defaults.")

    # 回退默认值（与历史脚本一致）
    defaults = {
        'npy_file': "/home/wangshuo/otdr/out/atten_weights/1724315190.682787.jpg_cross_attn_weight_5.npy",
        'base_image_dir': "/data20t/wangshuo/IR_Drivable/OTDR/test/images/xts_5",
        'patch_idx': 795,
        'frames': None,  # None 表示从数据推断
        'spatial_h': 32,
        'spatial_w': 40,
        'output_root': "/home/wangshuo/otdr/out/atten_vis",
    }

    # 组装最终配置
    def pick(key):
        if isinstance(cfg_values, dict) and key in cfg_values:
            return cfg_values[key]
        try:
            return getattr(cfg_values, key)
        except Exception:
            return defaults[key]

    return {
        'npy_file': pick('npy_file'),
        'base_image_dir': pick('base_image_dir'),
        'patch_idx': pick('patch_idx'),
        'frames': pick('frames'),
        'spatial_h': pick('spatial_h'),
        'spatial_w': pick('spatial_w'),
        'output_root': pick('output_root'),
    }

def extract_image_name(npy_file_path):
    """从.npy文件路径提取图片名称"""
    filename = os.path.basename(npy_file_path)
    # 移除 '_cross_attn_weight_3.npy' 后缀
    image_name = filename.replace('_cross_attn_weight_3.npy', '')
    return image_name

def find_image_path(image_name, base_dir):
    """在指定目录中查找对应的图片文件"""
    # 查找可能的图片格式
    for ext in ['.jpg', '.png', '.jpeg']:
        pattern = f"{base_dir}/**/*{image_name}{ext}"
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None

def annotate_patch_on_image(image_path, patch_idx, save_path, spatial_size=(32, 40), patch_size=16):
    """在原图上标注patch位置"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    
    print(f"Image size: {w_img} x {h_img}")
    print(f"Spatial size: {spatial_size}")
    print(f"Patch index: {patch_idx}")
    
    # 计算patch位置
    h, w = spatial_size
    patch_y, patch_x = divmod(patch_idx, w)  # 修正：patch_idx // w = row(y), patch_idx % w = col(x)
    
    print(f"Patch grid position: ({patch_x}, {patch_y})")
    
    # 计算在原图中的位置（特征图尺寸到原图尺寸的映射）
    x_start = int(patch_x * w_img / w)
    y_start = int(patch_y * h_img / h)
    x_end = int((patch_x + 1) * w_img / w)
    y_end = int((patch_y + 1) * h_img / h)
    
    # 确保坐标在图像范围内
    x_start = max(0, min(x_start, w_img-1))
    y_start = max(0, min(y_start, h_img-1))
    x_end = max(0, min(x_end, w_img-1))
    y_end = max(0, min(y_end, h_img-1))
    
    print(f"Rectangle coordinates: ({x_start}, {y_start}) to ({x_end}, {y_end})")
    print(f"Rectangle size: {x_end - x_start} x {y_end - y_start}")
    
    # 在图像上绘制红色方框
    cv2.rectangle(img_rgb, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
        
    # 保存标注图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(f'Query Patch {patch_idx} at ({patch_x}, {patch_y})')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Annotated image saved to: {save_path}")

def find_history_images(current_image_path, num_frames=4):
    """找到当前图片前的历史图片"""
    current_dir = os.path.dirname(current_image_path)
    current_name = os.path.basename(current_image_path)
    
    # 获取目录中所有图片并排序
    all_images = sorted(glob.glob(f"{current_dir}/*.jpg"))
    
    try:
        current_idx = all_images.index(current_image_path)
        # 获取前num_frames张图片
        history_images = []
        for i in range(num_frames):
            hist_idx = current_idx - num_frames + i
            if hist_idx >= 0:
                history_images.append(all_images[hist_idx])
            else:
                history_images.append(all_images[0])  # 用第一张图片填充
        return history_images
    except ValueError:
        print(f"Current image not found in directory: {current_image_path}")
        return []

def create_attention_overlay(image_path, attention_map, save_path, alpha=0.6, global_min=None, global_max=None):
    """创建注意力热图与原图的叠加"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    
    # 将注意力图调整到原图尺寸
    attention_resized = cv2.resize(attention_map, (w_img, h_img))
    
    # 使用全局归一化参数（如果提供）
    if global_min is not None and global_max is not None:
        attention_norm = (attention_resized - global_min) / (global_max - global_min)
    else:
        # 如果没有提供全局参数，使用局部归一化
        attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    
    # 确保归一化值在[0,1]范围内
    attention_norm = np.clip(attention_norm, 0, 1)
    
    # 创建热图
    heatmap = plt.cm.viridis(attention_norm)[:,:,:3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加图像
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(overlay)
    ax.axis('off')
    
    # 添加颜色条
    if global_min is not None and global_max is not None:
        # 创建颜色条映射
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                 norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=10)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention overlay saved to: {save_path}")

if __name__ == "__main__":
    # 仅接收配置文件路径的命令行参数
    parser = argparse.ArgumentParser(description="Visualize attention using settings from a config file")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to vis_atten config file')
    _args = parser.parse_args()

    # 从配置文件读取参数
    cfg = _load_vis_cfg(_args.config)

    patch_idx = int(cfg['patch_idx'])
    npy_file = str(cfg['npy_file'])
    image_name = extract_image_name(npy_file)
    print(f"Image name: {image_name}")
    base_image_dir = str(cfg['base_image_dir'])
    output_dir = f"{cfg['output_root']}/{os.path.splitext(image_name)[0]}/"

    # 2. 找到对应的图片路径
    image_path = find_image_path(os.path.splitext(image_name)[0], base_image_dir)
    if not image_path:
        print(f"Image not found for: {image_name}")
        exit(1)
    print(f"Found image: {image_path}")

    # 3. 标注patch位置并保存
    annotated_save_path = f"{output_dir}/image_mark_{patch_idx}.png"
    annotate_patch_on_image(image_path, patch_idx, annotated_save_path)

    # 4. 加载注意力权重
    attn_weight = np.load(npy_file)
    print(f"attn_weight.shape: {attn_weight.shape}")
    patch_attn = attn_weight[0][0][patch_idx]

    # 5. 根据配置/数据确定历史帧数并重塑
    spatial_h, spatial_w = int(cfg['spatial_h']), int(cfg['spatial_w'])
    per_frame_size = spatial_h * spatial_w
    total_size = int(np.prod(patch_attn.shape))
    if total_size % per_frame_size != 0:
        print(f"Error: attention vector size {total_size} is not divisible by H*W={per_frame_size}.")
        exit(1)
    inferred_frames = total_size // per_frame_size
    history_frames = int(cfg['frames']) if cfg['frames'] is not None else inferred_frames
    if history_frames != inferred_frames:
        print(f"Warning: frames={history_frames} does not match inferred {inferred_frames}. Using inferred value.")
        history_frames = inferred_frames

    attn_frames = patch_attn.reshape(history_frames, spatial_h, spatial_w)

    # 6. 计算全局min/max用于统一归一化
    global_min = attn_frames.min()
    global_max = attn_frames.max()
    print(f"Frames: {history_frames}, Spatial: ({spatial_h}, {spatial_w})")
    print(f"Global attention range: [{global_min:.6f}, {global_max:.6f}]")

    # 7. 找到历史图片
    history_images = find_history_images(image_path, history_frames)
    print(f"Found {len(history_images)} history images")

    # 8. 创建注意力叠加图（使用全局归一化）
    for i, (hist_img_path, attn_map) in enumerate(zip(history_images, attn_frames)):
        overlay_save_path = f"{output_dir}/overlays/patch_{patch_idx}_frame_T-{history_frames - i}_overlay.png"
        create_attention_overlay(hist_img_path, attn_map, overlay_save_path,
                                 )