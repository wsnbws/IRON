import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import mmcv
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import get_root_logger
from mmseg.core import DistEvalHook, EvalHook
import numpy as np
import random
from collections import OrderedDict
import json
from mmcv.runner import build_optimizer
import torch.optim.lr_scheduler as lr_scheduler
import math

DEBUG = False

class CustomTrainer:
    """自定义分布式训练器"""
    
    def __init__(self, cfg, distributed=False, local_rank=0):
        self.cfg = cfg
        self.distributed = distributed
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # 初始化分布式环境
        if distributed:
            self.rank, self.world_size = get_dist_info()
        else:
            self.rank = 0
            self.world_size = 1
            
        # 设置随机种子
        self._set_random_seed(cfg.seed)
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 构建模型
        self.model = self._build_model()
        
        # 构建数据集和数据加载器
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 构建优化器
        self.optimizer = self._build_optimizer()
        
        # 构建学习率调度器
        self.lr_scheduler = self._build_lr_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.current_iter = 0
        self.best_metric = 0.0
        
        # 检查点路径
        self.checkpoint_dir = os.path.join(cfg.work_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _set_random_seed(self, seed):
        """设置随机种子"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if self.distributed:
                torch.cuda.set_device(self.local_rank)
                
    def _init_logger(self):
        """初始化日志"""
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=self.cfg.log_level)
        return logger
        
    def _build_model(self):
        """构建模型"""
        model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )
        
        # 移动到设备
        model = model.to(self.device)
        
        # 分布式包装
        if self.distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.cfg.get('find_unused_parameters', False)
            )
            
        return model
        
    def _build_dataloaders(self):
        """构建数据加载器"""
        # 训练数据集
        train_dataset = build_dataset(self.cfg.data.train)
        
        # 验证数据集
        val_dataset = build_dataset(self.cfg.data.val, dict(test_mode=True))
        
        # 构建数据加载器
        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=True,
            seed=self.cfg.seed,
            drop_last=True
        )
        
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )
        
        return train_loader, val_loader
        
    def _build_optimizer(self):
        """构建优化器"""
        optimizer = build_optimizer(self.model, self.cfg.optimizer)
        return optimizer
        
    def _build_lr_scheduler(self):
        """构建学习率调度器"""
        warmup_steps = 4 * 187
        total_steps = 12000
        
        def lr_lambda(current_step):
            base_lr_factor = 1e-6 / self.cfg.optimizer['lr']  # 起始比例
            if current_step < warmup_steps:
                return base_lr_factor + (1 - base_lr_factor) * (current_step / warmup_steps)
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler
        
    def train_epoch(self):

        self.model.train() 
        for batch_idx, data_batch in enumerate(self.train_loader):

            data_batch = {'img': data_batch['img'].data[0].cuda(), 'gt_semantic_seg': data_batch['gt_semantic_seg'].data[0].cuda(), 'img_metas': data_batch['img_metas'].data[0]}
            if torch.cuda.current_device() == 0 and DEBUG == True:
                print(f"img: {data_batch['img'].shape}")
                print(f"gt_semantic_seg: {data_batch['gt_semantic_seg'].shape}")
                print(f"img_metas: {data_batch['img_metas']}")

            for t in range(int(len(data_batch['img_metas'][0]['frame_timestamps']))):
                losses = self.model(**data_batch, step=t)
                loss = losses["decode.loss_seg_final"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.lr_scheduler.step()
            
            if self.rank == 0 and batch_idx != 0 and (batch_idx % self.cfg.log_config.interval == 0 or batch_idx == len(self.train_loader) - 1):
                self.logger.info(
                    f'epoch: {self.current_epoch}, '
                    f'iter: {self.current_iter}, '
                    f'batch: {batch_idx}/{len(self.train_loader)}, '
                    f'train_seg_loss: {losses["decode.loss_seg_final"].item():.4f}, '
                    f'train_seg_acc: {losses["decode.acc_seg_final"].item():.4f}, '
                    f'lr: {self.optimizer.param_groups[-1]["lr"]:.4e} '
                )
            self.current_iter += 1
        
    def validate(self):

        self.model.eval()
        results = []
        with torch.no_grad():
            for data_batch in self.val_loader:
                data_batch = {'img': [_.cuda() for _ in data_batch['img']], 'img_metas': [_.data[0] for _ in data_batch['img_metas']], 'return_loss': False}
                if torch.cuda.current_device() == 0 and DEBUG == True:
                    print(f"img: {data_batch['img']}")
                    print(f"img_metas: {data_batch['img_metas']}")
                with torch.no_grad():
                    result = self.model(**data_batch)
                    results.extend(result)
            eval_metrics = self.val_loader.dataset.evaluate(results, logger=self.logger)
        return eval_metrics
        
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        if self.rank != 0:
            return
            
        checkpoint = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_metric': self.best_metric,
            'cfg': self.cfg
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'Best model saved to {best_path}')
            
        # 定期保存检查点
        if self.current_epoch % self.cfg.checkpoint_config.interval == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{self.current_epoch}.pth')
            torch.save(checkpoint, epoch_path)
            
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f'Checkpoint {checkpoint_path} not found')
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载学习率调度器状态
        if self.lr_scheduler and checkpoint.get('lr_scheduler_state_dict'):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        # 恢复训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iter', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        self.logger.info(f'Loaded checkpoint from {checkpoint_path}')
        
    def train(self, max_epochs=None, max_iters=None):
        """主训练循环"""
        self.logger.info('Starting training...')
        
        # 加载预训练模型或恢复训练
        if self.cfg.get('load_from'):
            self.load_checkpoint(self.cfg.load_from)
        elif self.cfg.get('resume_from'):
            self.load_checkpoint(self.cfg.resume_from)
            
        start_epoch = self.current_epoch
        
        # 确定训练轮数
        if max_epochs is None:
            max_epochs = self.cfg.get('total_epochs', 40)
            
        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch
            
            # 设置分布式采样器的epoch
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
                
            self.train_epoch()

            if (epoch + 1) % self.cfg.val_epoch == 0:
                eval_metrics = self.validate()

                if self.rank == 0:
                    self.logger.info(
                        f'Epoch {epoch + 1}: , '
                        f'mIoU: {eval_metrics["mIoU"]:.4f}, '
                        f'mAcc: {eval_metrics["mAcc"]:.4f}, '
                        f'aAcc: {eval_metrics["aAcc"]:.4f}, '
                    )
                    
                    is_best = eval_metrics['mIoU'] > self.best_metric
                    if is_best:
                        self.best_metric = eval_metrics['mIoU']
                        self.save_checkpoint(is_best=is_best)
                    
            if self.distributed:
                dist.barrier()
                
        self.logger.info('Training completed!')
