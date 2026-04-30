import argparse
import os
import sys
import time

sys.path.append("..")
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import DictAction

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from dataset import drivable
from backbone import convmae


def parse_args():
    parser = argparse.ArgumentParser(description='Test model FPS')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def single_gpu_fps_test(model, data_loader):
    """Test FPS with single GPU.
    
    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        
    Returns:
        float: Average FPS.
    """
    model.eval()
    dataset = data_loader.dataset
    total_time = 0.0
    total_samples = 0
    
    print(f"Testing FPS on {len(dataset)} samples...")
    
    # 预热GPU
    print("Warming up...")
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= 10:  # 预热10个batch
                break
            _ = model(return_loss=False, **data)
    
    # 正式测试
    print("Starting FPS test...")
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            batch_size = data['img'][0].size(0)
            
            # 记录开始时间
            torch.cuda.synchronize()  # 确保GPU操作完成
            start_time = time.time()
            
            # 模型推理
            result = model(return_loss=False, **data)
            
            # 记录结束时间
            torch.cuda.synchronize()  # 确保GPU操作完成
            end_time = time.time()
            
            # 累计时间和样本数
            total_time += (end_time - start_time)
            total_samples += batch_size
            
            # 打印进度
            if (i + 1) % 50 == 0:
                current_fps = total_samples / total_time
                print(f"Processed {i+1}/{len(data_loader)} batches, "
                      f"Current FPS: {current_fps:.2f}")
    
    avg_fps = total_samples / total_time
    return avg_fps


def main():
    args = parse_args()
    
    # 加载配置
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # 设置cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    
    # 构建数据加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    # 构建模型并加载权重
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    
    # 使用单GPU
    model = MMDataParallel(model, device_ids=[0])
    
    # 测试FPS
    avg_fps = single_gpu_fps_test(model, data_loader)
    
    print(f"\n=== FPS Test Results ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference time per image: {1000/avg_fps:.2f} ms")


if __name__ == '__main__':
    main()