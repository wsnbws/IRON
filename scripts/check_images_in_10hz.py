#!/usr/bin/env python3
"""
检查 /public/home/wangshuo/DrivableTask/data/Video/videos/validation 下所有目录的图片
是否能在 /public/home/public/IR_Drivable/image_10hz 下找到（基于文件名匹配）。

使用方法：
  python3 /public/home/wangshuo/DrivableTask/scripts/check_images_in_10hz.py \
    --source /public/home/wangshuo/DrivableTask/data/Video/videos/validation \
    --target /public/home/public/IR_Drivable/image_10hz \
    --output /public/home/wangshuo/DrivableTask/scripts/out_validation_vs_10hz.csv

说明：
- 匹配逻辑为“同名文件名”在目标树任意位置出现即视为存在。
- 统计会按 source 根目录下的“一级子目录”进行聚合输出。
- 可通过 --exts 自定义图片后缀，默认：jpg,jpeg,png,bmp,tiff,tif
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def is_image_file(filename: str, allowed_exts: Set[str]) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip(".") in allowed_exts


def walk_files(root_dir: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            yield os.path.join(dirpath, fname)


def index_target_by_basename(target_root: str, allowed_exts: Set[str]) -> Dict[str, List[str]]:
    basename_to_paths: Dict[str, List[str]] = defaultdict(list)
    for path in walk_files(target_root):
        if is_image_file(path, allowed_exts):
            base = os.path.basename(path)
            basename_to_paths[base].append(path)
    return basename_to_paths


def collect_source_images(source_root: str, allowed_exts: Set[str]) -> List[str]:
    images: List[str] = []
    for path in walk_files(source_root):
        if is_image_file(path, allowed_exts):
            images.append(path)
    return images


def group_by_top_level_dir(paths: Sequence[str], source_root: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    src_root_norm = os.path.normpath(source_root)
    for full_path in paths:
        rel = os.path.relpath(full_path, src_root_norm)
        parts = rel.split(os.sep)
        top = parts[0] if len(parts) > 1 else "__root__"
        groups[top].append(full_path)
    return groups


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def parse_exts(exts_arg: str) -> Set[str]:
    normalized = {ext.strip().lower().lstrip(".") for ext in exts_arg.split(",") if ext.strip()}
    return {e for e in normalized if e}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="验证图片是否存在于目标目录（基于文件名匹配）")
    parser.add_argument(
        "--source",
        default="/public/home/wangshuo/DrivableTask/data/Video/videos/validation",
        help="源目录（包含若干子目录与图片）",
    )
    parser.add_argument(
        "--target",
        default="/public/home/public/IR_Drivable/image_10hz",
        help="目标目录（大图像库，任意位置出现同名即视为存在）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选：输出 CSV 文件路径，记录每张图是否存在及候选位置",
    )
    parser.add_argument(
        "--exts",
        default="jpg,jpeg,png,bmp,tiff,tif",
        help="逗号分隔的图片扩展名（不带点）",
    )

    args = parser.parse_args(argv)
    source_root = os.path.abspath(args.source)
    target_root = os.path.abspath(args.target)
    allowed_exts = parse_exts(args.exts)

    if not os.path.isdir(source_root):
        print(f"[错误] 源目录不存在: {source_root}")
        return 2
    if not os.path.isdir(target_root):
        print(f"[错误] 目标目录不存在: {target_root}")
        return 2

    print("[1/3] 索引目标目录下的所有图片（按文件名）……")
    basename_to_paths = index_target_by_basename(target_root, allowed_exts)
    total_target_images = sum(len(v) for v in basename_to_paths.values())
    print(f"目标目录去重前图片总数（按文件）：{total_target_images}")
    print(f"目标目录去重后唯一文件名数：{len(basename_to_paths)}")

    print("[2/3] 收集源目录下的所有图片……")
    source_images = collect_source_images(source_root, allowed_exts)
    print(f"源目录图片总数：{len(source_images)}")

    print("[3/3] 按源目录一级子目录统计存在/缺失情况……\n")
    groups = group_by_top_level_dir(source_images, source_root)

    grand_total = 0
    grand_found = 0
    grand_missing = 0

    rows: List[Tuple[str, str, str, bool, int, str]] = []

    # 逐组统计
    for group_name in sorted(groups.keys()):
        items = groups[group_name]
        total = len(items)
        found = 0
        missing = 0
        for src_path in items:
            base = os.path.basename(src_path)
            candidates = basename_to_paths.get(base, [])
            exists = len(candidates) > 0
            if exists:
                found += 1
            else:
                missing += 1

            if args.output:
                rel_src = os.path.relpath(src_path, source_root)
                rows.append(
                    (
                        group_name,
                        rel_src,
                        base,
                        exists,
                        len(candidates),
                        " | ".join(candidates[:10]),  # 防止过长
                    )
                )

        grand_total += total
        grand_found += found
        grand_missing += missing

        rate = (found / total * 100.0) if total > 0 else 0.0
        print(
            f"目录: {group_name:30s} 总数: {total:6d}  找到: {found:6d}  缺失: {missing:6d}  命中率: {rate:6.2f}%"
        )

    print("\n总体统计：")
    overall_rate = (grand_found / grand_total * 100.0) if grand_total > 0 else 0.0
    print(
        f"总数: {grand_total:6d}  找到: {grand_found:6d}  缺失: {grand_missing:6d}  命中率: {overall_rate:6.2f}%"
    )

    if args.output:
        ensure_parent_dir(args.output)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "group", "source_rel_path", "filename", "exists_in_target",
                "num_candidates", "candidate_paths_sample"
            ])
            for row in rows:
                writer.writerow(row)
        print(f"\n明细 CSV 已写入: {os.path.abspath(args.output)}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


