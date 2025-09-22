import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner, get_dist_info

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
try:
    import apex
except:
    print('apex is not installed')
import torch.distributed as dist
from mmcv.runner.hooks.hook import Hook

           
class SingleGpuEvalHookWithBarrier(EvalHook):

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        # barrier BEFORE eval to align all ranks
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        # single-gpu evaluation on rank0
        super().after_train_iter(runner)
        # barrier AFTER eval to release other ranks
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        super().after_train_epoch(runner)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


class EvalBarrierHook(Hook):

    def __init__(self, interval=1, by_epoch=False):
        self.interval = interval
        self.by_epoch = by_epoch

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return   
        # clear local log buffer and disable logging for this step on non-rank0
        if hasattr(runner, 'log_buffer'):
            runner.log_buffer.clear()
            runner.log_buffer.ready = False
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # match rank0 pre-eval barrier
            dist.barrier()  # match rank0 post-eval barrier
            
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        if hasattr(runner, 'log_buffer'):
            runner.log_buffer.clear()
            runner.log_buffer.ready = False
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.barrier()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # use apex fp16 optimizer
    if cfg.optimizer_config.get("type", None) and cfg.optimizer_config["type"] == "DistOptimizerHook":
        if cfg.optimizer_config.get("use_fp16", False):
            # Move model to current device before apex initialization
            device = torch.cuda.current_device() if distributed else cfg.gpu_ids[0]
            model = model.cuda(device)
            model, optimizer = apex.amp.initialize(
                model, optimizer, opt_level="O1")
            for m in model.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        # Model is already on the correct device from apex initialization
        if not (cfg.optimizer_config.get("type", None) == "DistOptimizerHook" and cfg.optimizer_config.get("use_fp16", False)):
            model = model.cuda(torch.cuda.current_device())
        model = MMDistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = 'IterBasedRunner' not in cfg.runner['type']

        # If training is distributed but validation should be single-GPU,
        # build a non-distributed dataloader and only register EvalHook on rank 0
        single_gpu_val = bool(eval_cfg.get('single_gpu', False))
        val_dist = distributed and not single_gpu_val
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=val_dist,
            shuffle=False)

        if distributed and single_gpu_val:
            rank, _ = get_dist_info()
            interval = eval_cfg.get('interval', 1)
            by_epoch = eval_cfg.get('by_epoch', False)
            if rank == 0:
                runner.register_hook(SingleGpuEvalHookWithBarrier(val_dataloader, **eval_cfg))
            else:
                runner.register_hook(EvalBarrierHook(interval=interval, by_epoch=by_epoch))
        else:
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
