#!/usr/bin/env python3
"""
自定义分布式训练脚本
基于原有的train.py和train_api.py，提供更灵活的训练控制
"""

import argparse
import os
import os.path as osp
import sys
import time

# 添加项目路径
sys.path.append(".")
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import set_random_seed
from mmseg.utils import collect_env, get_root_logger
from custom_trainer import CustomTrainer
import backbone
import dataset
import head
import segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Distributed Training')
    
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录，用于保存日志和模型')
    parser.add_argument('--load-from', help='预训练模型路径')
    parser.add_argument('--resume-from', help='恢复训练的检查点路径')
    
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='使用的GPU数量 (仅适用于非分布式训练)'
    )
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='使用的GPU ID列表 (仅适用于非分布式训练)'
    )
    
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='是否设置CUDNN确定性选项'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='最大训练轮数'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=None,
        help='最大训练迭代数'
    )
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='作业启动器'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--master-port',
        type=int,
        default=None,
        help='分布式训练的主端口号'
    )
    
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='自定义选项'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='训练期间不进行验证'
    )
    
    args = parser.parse_args()
    
    # 设置本地rank环境变量
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        
    return args


def setup_environment(cfg, args):

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                               osp.splitext(osp.basename(args.config))[0])
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
        
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        
    return cfg


def init_distributed_training(args, cfg):
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        
    return distributed


def setup_logging(cfg, distributed):
    """设置日志"""
    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # 保存配置文件
    cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg.filename)))
    
    # 初始化日志
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    # 记录环境信息
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    
    # 记录基本信息
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    return logger, meta


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        
    cfg = setup_environment(cfg, args)
    
    distributed = init_distributed_training(args, cfg)
    
    logger, meta = setup_logging(cfg, distributed)
    
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    
    if distributed:
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank, world_size, local_rank = 0, 1, 0
        
    logger.info(f'Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}')
    
    trainer = CustomTrainer(
        cfg=cfg,
        distributed=distributed,
        local_rank=local_rank
    )

    try:
        trainer.train(
            max_epochs=args.max_epochs,
            max_iters=args.max_iters
        )
    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
        if distributed:
            torch.distributed.destroy_process_group()
    except Exception as e:
        logger.error(f'Training failed with error: {e}')
        if distributed:
            torch.distributed.destroy_process_group()
        raise
        
    logger.info('Training completed successfully!')


if __name__ == '__main__':
    main()
