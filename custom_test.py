#!/usr/bin/env python3
"""
单卡测试脚本 - 计算mIoU并可视化结果
基于custom_trainer.py的验证方法，确保推理和评估的一致性
集成自定义可视化器，支持分割掩码、预测点和置信度的可视化

使用方法:
python custom_test.py config/path/config.py checkpoints/model.pth --eval --show-dir ./vis_results

参数说明:
- config: 配置文件路径
- checkpoint: 模型检查点路径
- --eval: 计算评估指标 (可选)
- --show: 显示结果 (可选)
- --show-dir: 可视化结果保存目录 (可选)
- --device: 测试设备，默认cuda:0 (可选)
- --threshold: 二值化阈值，默认0.5 (可选)

可视化参数:
- --show-mask: 显示分割掩码 (默认True)
- --show-points: 显示预测点 (默认True)
- --show-confidence: 显示置信度 (默认True)
- --show-info-text: 显示坐标和置信度文本信息 (默认True)
- --mask-alpha: 掩码透明度，0.0-1.0 (默认0.5)
- --point-radius: 点的半径 (默认5)
"""

import argparse
import os
import os.path as osp
import sys
import mmcv
import torch
import numpy as np
from mmcv.image import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

# 添加项目路径
sys.path.append(".")

from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import get_root_logger
from mmseg.ops import resize

# 导入自定义模块
import backbone
import dataset
import head
import segmentor
from scripts.metrics import create_evaluator
from scripts.vis import SegmentationVisualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单卡模型测试')
    
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('checkpoint', help='模型检查点路径')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--show-dir', help='可视化结果保存目录')
    parser.add_argument('--eval', action='store_true', help='评估指标')
    parser.add_argument('--device', default='cuda:0', help='测试设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    
    # 可视化相关参数
    parser.add_argument('--show-mask', action='store_true', default=True, help='显示分割掩码')
    parser.add_argument('--show-points', action='store_true', default=False, help='显示预测点')
    parser.add_argument('--show-confidence', action='store_true', default=False, help='显示置信度')
    parser.add_argument('--show-info-text', action='store_true', default=False, help='显示坐标和置信度文本信息')
    parser.add_argument('--mask-alpha', type=float, default=0.5, help='掩码透明度')
    parser.add_argument('--point-radius', type=int, default=5, help='点的半径')
    
    return parser.parse_args()


def single_gpu_test(model, data_loader, show=False, out_dir=None, threshold=0.5, eval_mode=False, evaluator=None, visualizer=None):
    """单GPU测试函数，基于custom_trainer.py的验证方法，支持随推理随计算指标"""
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data_batch in enumerate(data_loader):
        # 使用与trainer相同的数据预处理方式
        imgs = [_.cuda() for _ in data_batch['img']]
        img_metas = [_.data[0] for _ in data_batch['img_metas']]
        
        with torch.no_grad():
            # 使用与trainer相同的forward_test方法
            seg_pred, confidence, points = model.forward_test(imgs, img_metas, threshold=threshold)
            seg_pred = seg_pred[0]

        if eval_mode and evaluator is not None:
            batch_gts = []
            for img_meta in img_metas[0]:
                gt_path = os.path.join(dataset.ann_dir, img_meta['ann']['seg_map'])
                gt_seg = mmcv.imread(gt_path, flag='unchanged', backend='pillow')
                assert gt_seg is not None, f"GT path not found: {gt_path}"
                batch_gts.append(gt_seg)
            
            batch_gts = np.stack(batch_gts, axis=0)
            evaluator.add_batch(seg_pred, batch_gts)

        if (show or out_dir) and visualizer is not None:
            img_tensor = imgs[0]
            img_metas_batch = img_metas[0]
            imgs_vis = tensor2imgs(img_tensor, **img_metas_batch[0]['img_norm_cfg'])
            assert len(imgs_vis) == len(img_metas_batch)

            for idx, (img, img_meta) in enumerate(zip(imgs_vis, img_metas_batch)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                
                img_show_bgr = img_show[..., ::-1]
                seg_mask = seg_pred[idx] if len(seg_pred.shape) > 2 else seg_pred
                batch_points = None
                batch_confidence = None
                
                points_show = None
                confidence_show = None
                if points is not None:
                    # points shape: (B, num_points, 2)
                    if isinstance(points, torch.Tensor):
                        points_np = points.cpu().numpy()
                    else:
                        points_np = points
                    
                    points_show = points_np[idx]
                
                if confidence is not None:
                    # confidence shape: (B, num_points)
                    if isinstance(confidence, torch.Tensor):
                        confidence_np = confidence.cpu().numpy()
                    else:
                        confidence_np = confidence
                    confidence_show = confidence_np[idx]

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                
                result_img = visualizer.display(
                    img=img_show_bgr,
                    seg_mask=seg_mask,
                    points=points_show,
                    confidence=confidence_show,
                    out_file=out_file
                )

        batch_size = len(img_metas)
        for _ in range(batch_size):
            prog_bar.update()
            
    return results


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 初始化日志
    logger = get_root_logger(log_level=cfg.log_level)
    
    # 构建数据集和数据加载器
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载检查点
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict["model_state_dict"], strict = False)
    print(f"missing: {missing}")
    print(f"unexpected: {unexpected}")
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    
    # 移动到指定设备
    model = model.to(args.device)
    model.eval()
    
    # 获取配置中的参数
    num_classes = getattr(cfg.model.decode_head, 'num_classes', 2)
    ignore_index = getattr(cfg, 'ignore_index', 255)
    threshold = getattr(cfg, 'test_threshold', args.threshold)
    
    logger.info(f'开始测试，共 {len(dataset)} 张图像')
    logger.info(f'模型类别数: {len(model.CLASSES)}')
    logger.info(f'使用设备: {args.device}')
    logger.info(f'二值化阈值: {threshold}')
    logger.info(f'忽略索引: {ignore_index}')
    
    # 创建可视化器
    visualizer = None
    if args.show or args.show_dir:
        # 创建输出目录
        if args.show_dir:
            mmcv.mkdir_or_exist(args.show_dir)
        
        # 配置可视化器
        visualizer = SegmentationVisualizer(
            show_mask=args.show_mask,
            show_points=args.show_points,
            show_confidence=args.show_confidence,
            show_info_text=args.show_info_text,
            palette=dataset.PALETTE if hasattr(dataset, 'PALETTE') else None,
            point_color=(0, 0, 255),  # 红色点
            point_radius=args.point_radius,
            mask_alpha=args.mask_alpha,
            confidence_range=(0.3, 1.0)
        )
        
        logger.info('可视化配置:')
        logger.info(f'  显示掩码: {args.show_mask}')
        logger.info(f'  显示点: {args.show_points}')
        logger.info(f'  显示置信度: {args.show_confidence}')
        logger.info(f'  显示信息文本: {args.show_info_text}')
        logger.info(f'  掩码透明度: {args.mask_alpha}')
        logger.info(f'  点半径: {args.point_radius}')
    
    # 执行测试
    if args.eval:
        # 创建增量评估器，随推理随计算指标
        logger.info('开始测试和评估...')
        evaluator = create_evaluator(
            num_classes=num_classes,
            ignore_index=ignore_index,
            metrics=['mIoU']
        )
        
        # 使用增量评估模式
        results = single_gpu_test(
            model, 
            data_loader, 
            show=args.show, 
            out_dir=args.show_dir,
            threshold=threshold,
            eval_mode=True,
            evaluator=evaluator,
            visualizer=visualizer
        )
        
        # 计算最终指标
        eval_metrics = evaluator.compute(logger=logger)
        
        logger.info('=' * 50)
        logger.info('测试结果:')
        logger.info(f'mIoU: {eval_metrics["mIoU"]:.4f}')
        logger.info(f'aAcc: {eval_metrics["aAcc"]:.4f}')
        
    else:
        # 仅进行推理，不计算指标
        logger.info('开始测试...')
        results = single_gpu_test(
            model, 
            data_loader, 
            show=args.show, 
            out_dir=args.show_dir,
            threshold=threshold,
            eval_mode=False,
            evaluator=None,
            visualizer=visualizer
        )
        
    if args.show_dir:
        logger.info(f'可视化结果已保存到: {args.show_dir}')
    
    logger.info('测试完成!')


if __name__ == '__main__':
    main()

# 使用示例:
# 
# 1. 基本测试（仅推理）:
# python custom_test.py config.py model.pth --show-dir ./results
#
# 2. 测试并评估:
# python custom_test.py config.py model.pth --eval --show-dir ./results
#
# 3. 自定义可视化设置:
# python custom_test.py config.py model.pth --show-dir ./results \
#   --show-mask --show-points --show-confidence --show-info-text \
#   --mask-alpha 0.6 --point-radius 8
#
# 4. 只显示掩码，不显示点和文本信息:
# python custom_test.py config.py model.pth --show-dir ./results \
#   --show-mask --no-show-points --no-show-confidence --no-show-info-text
#
# 5. 只显示点和坐标信息，不显示掩码:
# python custom_test.py config.py model.pth --show-dir ./results \
#   --no-show-mask --show-points --show-info-text
