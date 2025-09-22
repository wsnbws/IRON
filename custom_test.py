#!/usr/bin/env python3
"""
单卡测试脚本 - 计算mIoU并可视化结果
基于custom_trainer.py和test.py的功能，提供简洁的测试接口

使用方法:
python test_single.py config/path/config.py checkpoints/model.pth --show-dir ./vis_results

参数说明:
- config: 配置文件路径
- checkpoint: 模型检查点路径
- --show: 显示结果 (可选)
- --show-dir: 可视化结果保存目录 (可选)
- --device: 测试设备，默认cuda:0 (可选)
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单卡模型测试')
    
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('checkpoint', help='模型检查点路径')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--show-dir', help='可视化结果保存目录')
    parser.add_argument('--eval', action='store_true', help='评估指标')
    parser.add_argument('--device', default='cuda:0', help='测试设备')
    
    return parser.parse_args()


def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """单GPU测试函数，基于mmseg/apis/test.py的single_gpu_test"""
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data_batch in enumerate(data_loader):
        data = {'img': [_.cuda() for _ in data_batch['img']], 'img_metas': [_.data[0] for _ in data_batch['img_metas']], 'return_loss': False}
        with torch.no_grad():
            result = model(**data)

        # 可视化处理
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for idx, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file
                )

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        batch_size = len(data['img_metas'])
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
    checkpoint = model.load_state_dict(state_dict["model_state_dict"])
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    
    # 移动到指定设备
    model = model.to(args.device)
    model.eval()
    
    logger.info(f'开始测试，共 {len(dataset)} 张图像')
    logger.info(f'模型类别数: {len(model.CLASSES)}')
    logger.info(f'使用设备: {args.device}')
    
    # 执行测试
    results = single_gpu_test(
        model, 
        data_loader, 
        show=args.show, 
        out_dir=args.show_dir
    )
    
    if args.eval:
        # 计算评估指标
        logger.info('开始计算评估指标...')
        eval_metrics = dataset.evaluate(results, logger=logger)
        
        logger.info('=' * 50)
        logger.info('测试结果:')
        logger.info(f'mIoU: {eval_metrics["mIoU"]:.4f}')
        logger.info(f'mAcc: {eval_metrics["mAcc"]:.4f}') 
        logger.info(f'aAcc: {eval_metrics["aAcc"]:.4f}')

        if 'IoU' in eval_metrics:
            logger.info('\n各类别IoU:')
            for i, (cls_name, iou) in enumerate(zip(model.CLASSES, eval_metrics['IoU'])):
                logger.info(f'{cls_name}: {iou:.4f}')
        
    if args.show_dir:
        logger.info(f'可视化结果已保存到: {args.show_dir}')
    
    logger.info('测试完成!')


if __name__ == '__main__':
    main()
