import os
import re
import cv2
import argparse
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate videos from classified images')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('--output_dir', default='./videos', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=5, help='Frame rate for output videos')
    parser.add_argument('--video_format', default='mp4', choices=['mp4', 'avi'], 
                       help='Output video format')
    parser.add_argument('--single', action='store_true',
                       help='If set, create a single video from all images in the directory')
    parser.add_argument('--output_name', default=None,
                       help='Output filename (without extension) when --single is set; default: directory name')
    return parser.parse_args()


def extract_info_from_filename(filename):
    """
    从文件名提取前缀和时间戳
    例如: 20240510-221637-night-sunny-ITLQy6_1715351018.581172.jpg
    返回: ('20240510-221637-night-sunny-ITLQy6', 1715351018.581172)
    """
    if not filename.endswith('.jpg'):
        return None, None
    
    # 移除.jpg后缀
    name_without_ext = filename[:-4]
    
    # 查找最后一个下划线，分离前缀和时间戳
    last_underscore = name_without_ext.rfind('_')
    if last_underscore == -1:
        return None, None
    
    prefix = name_without_ext[:last_underscore]
    timestamp_str = name_without_ext[last_underscore + 1:]
    
    try:
        timestamp = float(timestamp_str)
        return prefix, timestamp
    except ValueError:
        return None, None


def classify_images(input_dir):
    """
    将图片按前缀分类
    返回: {prefix: [(timestamp, filepath), ...]}
    """
    image_groups = defaultdict(list)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jpg'):
            continue
            
        prefix, timestamp = extract_info_from_filename(filename)
        if prefix is not None and timestamp is not None:
            filepath = os.path.join(input_dir, filename)
            image_groups[prefix].append((timestamp, filepath))
        else:
            print(f"Warning: Cannot parse filename {filename}")
    
    # 按时间戳排序每组图片
    for prefix in image_groups:
        image_groups[prefix].sort(key=lambda x: x[0])
    
    return image_groups


def collect_images_in_dir(input_dir):
    """
    收集目录下所有.jpg图片，按解析出的时间戳排序；
    若无法解析时间戳，则按文件名排序。
    返回: [(timestamp or idx, filepath), ...]
    """
    items = []
    fallback = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jpg'):
            continue
        prefix, ts = extract_info_from_filename(filename)
        path = os.path.join(input_dir, filename)
        if ts is None:
            fallback.append(path)
        else:
            items.append((ts, path))

    items.sort(key=lambda x: x[0])
    # append non-parsable images sorted by name for stability
    if fallback:
        for p in sorted(fallback):
            items.append((len(items), p))
    return items


def create_video_from_images(image_list, output_path, fps=40):
    """
    从图片列表创建视频
    """
    if not image_list:
        print(f"No images to create video for {output_path}")
        return False
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_list[0][1])
    if first_image is None:
        print(f"Cannot read first image: {image_list[0][1]}")
        return False
    
    height, width, channels = first_image.shape
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Cannot create video writer for {output_path}")
        return False
    
    print(f"Creating video: {output_path}")
    print(f"  - Images: {len(image_list)}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {len(image_list)/fps:.2f} seconds")
    
    # 写入每一帧
    for i, (timestamp, image_path) in enumerate(image_list):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Cannot read image {image_path}")
            continue
        
        # 确保图片尺寸一致
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        # 显示进度
        if (i + 1) % 50 == 0 or i == len(image_list) - 1:
            print(f"  Progress: {i+1}/{len(image_list)} frames")
    
    out.release()
    print(f"Video saved: {output_path}\n")
    return True


def main():
    args = parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 单视频模式：把目录内所有图片拼成一个视频
    if args.single:
        print("Collecting images for single video...")
        image_list = collect_images_in_dir(args.input_dir)
        if not image_list:
            print("No images found to create video.")
            return
        name = args.output_name or os.path.basename(os.path.normpath(args.input_dir))
        output_filename = f"{name}.{args.video_format}"
        output_path = os.path.join(args.output_dir, output_filename)
        ok = create_video_from_images(image_list, output_path, args.fps)
        if ok:
            print(f"Single video saved: {output_path}")
        return

    # 分组模式：按前缀分组为多个视频
    print("Classifying images...")
    image_groups = classify_images(args.input_dir)
    if not image_groups:
        print("No valid images found!")
        return

    print(f"Found {len(image_groups)} image groups:")
    for prefix, images in image_groups.items():
        print(f"  - {prefix}: {len(images)} images")

    print("\nGenerating videos...")
    success_count = 0
    for prefix, image_list in image_groups.items():
        safe_prefix = re.sub(r'[^\w\-_]', '_', prefix)
        output_filename = f"{safe_prefix}.{args.video_format}"
        output_path = os.path.join(args.output_dir, output_filename)
        if create_video_from_images(image_list, output_path, args.fps):
            success_count += 1
    print(f"Successfully created {success_count}/{len(image_groups)} videos")
    print(f"Videos saved in: {args.output_dir}")


if __name__ == '__main__':
    main()