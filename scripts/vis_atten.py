import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path

def visualize_attention_4frames(attn_weight, patch_idx, save_path, spatial_size=(32, 40), history_frames=4):
    """
    可视化指定patch对4个历史帧的注意力分数，使用统一颜色条
    
    Args:
        attn_weight: 注意力权重 shape: (32*32*4,) 
        patch_idx: patch索引
        save_path: 保存路径
        spatial_size: 空间尺寸 (H, W)
        history_frames: 历史帧数量
    """
    h, w = spatial_size
    
    # 重塑为4帧格式 [4, H, W]
    attn_frames = attn_weight.reshape(history_frames, h, w)
    
    # 计算patch位置
    patch_x, patch_y = divmod(patch_idx, w)
    
    # 创建4帧可视化，统一颜色条
    fig, axes = plt.subplots(1, history_frames, figsize=(16, 4))
    
    # 计算全局min/max用于统一颜色条
    vmin, vmax = attn_frames.min(), attn_frames.max()
    
    for i in range(history_frames):
        im = axes[i].imshow(attn_frames[i], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Frame T-{history_frames-i}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # 标记最高注意力位置
        max_pos = np.unravel_index(np.argmax(attn_frames[i]), attn_frames[i].shape)
        axes[i].plot(max_pos[1], max_pos[0], 'r*', markersize=10)
    
    # 添加统一颜色条
    # plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    
    plt.suptitle(f'Attention from Patch {patch_idx} ({patch_x}, {patch_y})', fontsize=14)
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {save_path}")

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

def create_attention_overlay(image_path, attention_map, save_path, alpha=0.6):
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
    
    # 归一化注意力图
    attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    
    # 创建热图
    heatmap = plt.cm.viridis(attention_norm)[:,:,:3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加图像
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention overlay saved to: {save_path}")

if __name__ == "__main__":
    # 配置参数
    patch_idx = 795
    npy_file = "/public/home/wangshuo/DrivableTask/out/DrivableSeg/deform_atten/v3_2/atten_vis/saved_weights/20240822-xts-day-5-KAAtil_1724315343.083632.jpg_cross_attn_weight_3.npy"
    image_name = extract_image_name(npy_file) 
    print(f"Image name: {image_name}")
    base_image_dir = "/public/home/public/IR_Drivable/baseline/images/test/xts_video/20240822-xts-day-5-KAAtil"
    output_dir = f"/public/home/wangshuo/DrivableTask/out/DrivableSeg/deform_atten/v3_2/atten_vis/atten_maps/{os.path.splitext(image_name)[0]}/"
    
    # 2. 找到对应的图片路径
    image_path = find_image_path(os.path.splitext(image_name)[0], base_image_dir)
    if not image_path:
        print(f"Image not found for: {image_name}")
        exit()
    print(f"Found image: {image_path}")
    
    # 3. 标注patch位置并保存
    annotated_save_path = f"{output_dir}/image_mark_{patch_idx}.png"
    annotate_patch_on_image(image_path, patch_idx, annotated_save_path)
    
    # 4. 加载注意力权重
    attn_weight = np.load(npy_file)
    print(f"attn_weight.shape: {attn_weight.shape}")
    patch_attn = attn_weight[0][0][patch_idx]  # (32*40*4)
    
    # 5. 重塑为4帧格式
    attn_frames = patch_attn.reshape(4, 32, 40)
    
    # 6. 找到历史图片
    history_images = find_history_images(image_path, 4)
    print(f"Found {len(history_images)} history images")
    
    # 7. 创建注意力叠加图
    for i, (hist_img_path, attn_map) in enumerate(zip(history_images, attn_frames)):
        overlay_save_path = f"{output_dir}/overlays/patch_{patch_idx}_frame_T-{4-i}_overlay.png"
        create_attention_overlay(hist_img_path, attn_map, overlay_save_path)
    
    