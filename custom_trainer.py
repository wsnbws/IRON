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
    
    def __init__(self, cfg, distributed=False, local_rank=0):
        self.cfg = cfg
        self.distributed = distributed
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        self.num_frames = cfg.data.train.get('num_frames', 8)

        if distributed:
            self.rank, self.world_size = get_dist_info()
        else:
            self.rank = 0
            self.world_size = 1
            
        self._set_random_seed(cfg.seed)

        self.logger = self._init_logger()
        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self.current_epoch = 0
        self.current_iter = 0
        self.best_metric = 0.0
    
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

        model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )
        
        model = model.to(self.device)
        
        if self.distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.cfg.get('find_unused_parameters', False)
            )
            
        return model
        
    def _build_dataloaders(self):

        train_dataset = build_dataset(self.cfg.data.train) 
        val_dataset = build_dataset(self.cfg.data.val, dict(test_mode=True))
        
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

        optimizer = build_optimizer(self.model, self.cfg.optimizer)
        return optimizer
        
    def _build_lr_scheduler(self):

        num_train_data = len(self.train_loader.dataset)
        samples_per_gpu = self.cfg.data.get('samples_per_gpu', 1)
        total_batch_size = samples_per_gpu * self.world_size
        batches_per_epoch = (num_train_data + total_batch_size - 1) // total_batch_size
        total_epochs = self.cfg.get('total_epochs', 10)
        total_iters = batches_per_epoch * total_epochs
        # 保存总迭代数供后续按iter/epoch验证逻辑判断使用
        self.total_iters = total_iters
        warmup_ratio = self.cfg.lr_config.get('warmup_ratio', 1e-6) if hasattr(self.cfg, 'lr_config') else 1e-6
        # 容错处理：未设置或为0则认为不启用warmup
        # warmup_iters = int((self.cfg.lr_config.get('warmup_iters', 0) if hasattr(self.cfg, 'lr_config') else 0) or 0)
        warmup_iters = int(float(self.cfg.lr_config.get('warmlen_ratio', 0)) * total_iters)
        
        if self.rank == 0:
            self.logger.info(f"Dataset size: {num_train_data}")
            self.logger.info(f"Samples per GPU: {samples_per_gpu}")
            self.logger.info(f"World size: {self.world_size}")
            self.logger.info(f"Total batch size: {total_batch_size}")
            self.logger.info(f"Batches per epoch: {batches_per_epoch}")
            self.logger.info(f"Total epochs: {total_epochs}")
            self.logger.info(f"Total iters: {total_iters}")
            self.logger.info(f"Warmup iters: {warmup_iters}")
            self.logger.info(f"Warmup ratio: {warmup_ratio}")
        
        def lr_lambda(current_iter):
            if warmup_iters > 0 and current_iter < warmup_iters:
                return warmup_ratio + (1.0 - warmup_ratio) * (current_iter / float(max(1, warmup_iters)))
            else:
                progress = float(current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler
        
    def _reduce_mean(self, tensor: torch.Tensor):
        
        if not self.distributed or not dist.is_initialized():
            return tensor
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.world_size
        return reduced

    def _save_best_checkpoint(self):
        """仅保存当前最优模型（按mIoU判定），文件名为best.pth；仅在rank0执行。"""
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
        best_path = os.path.join(self.checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        self.logger.info(f'Best model updated -> {best_path} (mIoU={self.best_metric:.4f})')

    def _update_best_and_maybe_save(self, eval_metrics: dict):
        """根据评估结果更新最优mIoU并在提升时保存best.pth；仅rank0触发保存。"""
        if not isinstance(eval_metrics, dict):
            return
        new_miou = eval_metrics.get('mIoU', None)
        if new_miou is None:
            return
        if new_miou > self.best_metric:
            self.best_metric = new_miou
            self._save_best_checkpoint()
    
    def _is_iter_time(self, interval: int) -> bool:
        """是否到达指定的迭代间隔（以1-based迭代编号判断）。"""
        return bool(interval) and interval > 0 and (((self.current_iter + 1) % interval) == 0)

    def _maybe_validate_and_checkpoint_iter(self):
        """按迭代间隔执行验证与保存（仅rank0做验证与保存）。"""
        eval_interval = None
        if hasattr(self.cfg, 'evaluation') and isinstance(self.cfg.evaluation, dict):
            eval_interval = self.cfg.evaluation.get('interval', None)
        ckpt_interval = None
        ckpt_by_epoch = True
        if hasattr(self.cfg, 'checkpoint_config') and isinstance(self.cfg.checkpoint_config, dict):
            ckpt_interval = self.cfg.checkpoint_config.get('interval', None)
            ckpt_by_epoch = self.cfg.checkpoint_config.get('by_epoch', True)

        # 按iter验证
        if eval_interval and self._is_iter_time(eval_interval):
            if self.distributed and dist.is_initialized():
                dist.barrier()
            eval_metrics = self.validate()
            if self.rank == 0:
                if isinstance(eval_metrics, dict):
                    self.logger.info(
                        f'Iter {self.current_iter + 1}: '
                        f'mIoU: {eval_metrics.get("mIoU", 0.0):.4f}, '
                        f'mAcc: {eval_metrics.get("mAcc", 0.0):.4f}, '
                        f'aAcc: {eval_metrics.get("aAcc", 0.0):.4f}'
                    )
                    self._update_best_and_maybe_save(eval_metrics)
            if self.distributed and dist.is_initialized():
                dist.barrier()
     
    def train_epoch(self):

        self.model.train() 
        
        for batch_idx, data_batch in enumerate(self.train_loader):
            data_batch = {'img': data_batch['img'].data[0].cuda(), 'gt_semantic_seg': data_batch['gt_semantic_seg'].data[0].cuda(), 'img_metas': data_batch['img_metas'].data[0]}
            if self.rank == 0 and DEBUG == True:
                print(f"img: {data_batch['img'].shape}")
                print(f"gt_semantic_seg: {data_batch['gt_semantic_seg'].shape}")
                print(f"img_metas: {data_batch['img_metas']}")

            batch_losses = {
                'decode.loss_seg_final': 0.0,
                'decode.loss_seg_hist': 0.0,
                'decode.acc_seg_final': 0.0,
                'decode.acc_seg_hist': 0.0
            }
        
            num_frames_in_batch = self.cfg.data.train.get('num_frames', 1)
            self.optimizer.zero_grad()
            
            for t in range(num_frames_in_batch):
                losses = self.model(**data_batch, step=t)
                loss = losses["decode.loss_seg_final"] + losses["decode.loss_seg_hist"]
                for key in batch_losses.keys():
                    batch_losses[key] += losses[key].item()

                loss.backward()
    
            self.optimizer.step()
            self.lr_scheduler.step()
            
            for key in batch_losses.keys():
                batch_losses[key] /= num_frames_in_batch

            # 多卡平均：将当前进程的batch指标在所有GPU之间做平均
            metrics_tensor = torch.tensor([
                batch_losses['decode.loss_seg_final'],
                batch_losses['decode.loss_seg_hist'],
                batch_losses['decode.acc_seg_final'],
                batch_losses['decode.acc_seg_hist'],
            ], device=self.device, dtype=torch.float32)

            metrics_tensor = self._reduce_mean(metrics_tensor)

            batch_losses['decode.loss_seg_final'] = metrics_tensor[0].item()
            batch_losses['decode.loss_seg_hist'] = metrics_tensor[1].item()
            batch_losses['decode.acc_seg_final'] = metrics_tensor[2].item()
            batch_losses['decode.acc_seg_hist'] = metrics_tensor[3].item()
            
            if self.rank == 0 and batch_idx != 0 and (batch_idx % self.cfg.log_config.interval == 0 or batch_idx == len(self.train_loader) - 1):
                self.logger.info(
                    f'epoch: {self.current_epoch + 1}, '
                    f'iter: {self.current_iter + 1}, '
                    f'batch: {batch_idx + 1}/{len(self.train_loader)}, '
                    f'avg_train_seg_loss: {batch_losses["decode.loss_seg_final"]:.4f}, '
                    f'avg_train_seg_acc: {batch_losses["decode.acc_seg_final"]:.4f}, '
                    f'avg_train_seg_hist_loss: {batch_losses["decode.loss_seg_hist"]:.4f}, '
                    f'avg_train_seg_hist_acc: {batch_losses["decode.acc_seg_hist"]:.4f}, '
                    f'lr: {self.optimizer.param_groups[-1]["lr"]:.4e} '
                )
            # 迭代级别的验证与保存
            self._maybe_validate_and_checkpoint_iter()
            self.current_iter += 1
        
    def validate(self):

        # if self.distributed and self.rank != 0:
        #     return None

        self.model.eval()
        results = []
        with torch.no_grad():
            for data_batch in self.val_loader:
                data_batch = {'img': [_.cuda() for _ in data_batch['img']], 'img_metas': [_.data[0] for _ in data_batch['img_metas']], 'return_loss': False}
                if self.rank == 0 and DEBUG == True:
                    print(f"img: {data_batch['img']}")
                    print(f"img_metas: {data_batch['img_metas']}")
                with torch.no_grad():
                    result = self.model(**data_batch)
                    results.extend(result)
            eval_metrics = self.val_loader.dataset.evaluate(results, logger=self.logger)
        self.model.train()
        return eval_metrics
        
    def save_checkpoint(self, is_best=False):
        """仅在is_best=True时保存best.pth；否则不保存。仅rank0执行。"""
        if self.rank != 0:
            return
        if not is_best:
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
        best_path = os.path.join(self.checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        self.logger.info(f'Best model saved to {best_path} (mIoU={self.best_metric:.4f})')
            
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

        self.logger.info('Starting training...')
        
        if self.cfg.get('load_from'):
            self.load_checkpoint(self.cfg.load_from)
        elif self.cfg.get('resume_from'):
            self.load_checkpoint(self.cfg.resume_from)
            
        start_epoch = self.current_epoch
        
        max_epochs = self.cfg.get('total_epochs', 40)
            
        # 若设置了evaluation.interval，则默认采用按iter验证；
        # 但当interval大于总iter数时，回退到按epoch验证，避免验证被跳过
        use_epoch_val = True
        if hasattr(self.cfg, 'evaluation') and isinstance(self.cfg.evaluation, dict):
            _eval_interval = self.cfg.evaluation.get('interval', None)
            if _eval_interval and _eval_interval > 0:
                # 正常采用按iter验证
                use_epoch_val = False
                # 如果interval过大，回退到按epoch验证
                try:
                    if hasattr(self, 'total_iters') and _eval_interval > self.total_iters:
                        use_epoch_val = True
                except Exception:
                    pass

        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch
            
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
                
            self.train_epoch()

            if use_epoch_val and ((self.current_epoch + 1) % self.cfg.val_epoch == 0):
                eval_metrics = self.validate()

                if self.rank == 0 and isinstance(eval_metrics, dict):
                    self.logger.info(
                        f'Epoch {self.current_epoch + 1}: , '
                        f'mIoU: {eval_metrics.get("mIoU", 0.0):.4f}, '
                        f'mAcc: {eval_metrics.get("mAcc", 0.0):.4f}, '
                        f'aAcc: {eval_metrics.get("aAcc", 0.0):.4f}, '
                    )
                    # 仅当变优时保存best
                    self._update_best_and_maybe_save(eval_metrics)
                    
            if self.distributed:
                dist.barrier()
                
        self.logger.info('Training completed!')
