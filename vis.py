import os
import cv2
import numpy as np
import glob
from pathlib import Path

import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor
from backbone import convmae

class SegmentationInference:
    def __init__(self, config_path, checkpoint_path, palette, device='cuda:0'):
        """
        初始化分割模型
        
        Args:
            config_path (str): 配置文件路径
            checkpoint_path (str): 模型权重文件路径
            palette (list): 调色板，每个元素是[R, G, B]格式的颜色
            device (str): 推理设备
        """
        self.device = device
        self.palette = palette
        
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # 加载配置文件
        print("Loading config...")
        self.cfg = mmcv.Config.fromfile(config_path)
        
        # 构建模型
        print("Building model...")
        self.model = build_segmentor(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        
        # 将配置附加到模型对象上
        self.model.cfg = self.cfg
        
        # 加载权重
        print("Loading checkpoint...")
        checkpoint = load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        
        # 如果使用CPU推理，需要将SyncBatchNorm转换为BatchNorm
        if device == 'cpu':
            print("Converting SyncBatchNorm to BatchNorm for CPU inference...")
            self.model = self._convert_sync_batchnorm(self.model)
        
        # 移动到指定设备
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def _convert_sync_batchnorm(self, module):
        """
        将SyncBatchNorm转换为BatchNorm，用于CPU推理
        
        Args:
            module: 要转换的模块
            
        Returns:
            转换后的模块
        """
        module_output = module
        if isinstance(module, torch.nn.SyncBatchNorm):
            module_output = torch.nn.BatchNorm2d(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, self._convert_sync_batchnorm(child))
        del module
        return module_output
    
    def create_colored_mask(self, result):
        """
        将分割结果转换为彩色掩码
        
        Args:
            result (np.ndarray): 分割结果
        
        Returns:
            np.ndarray: 彩色分割掩码
        """
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            if label < len(self.palette):
                color_seg[result == label, :] = color
        return color_seg
    
    def inference_single_image(self, img):
        """
        对单张图像进行推理的替代方法
        
        Args:
            img (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 分割结果
        """
        with torch.no_grad():
            # 预处理图像
            data = dict(img=img)
            data = self.cfg.data.test.pipeline(data)
            data = mmcv.parallel.collate([data], samples_per_gpu=1)
            
            if next(self.model.parameters()).is_cuda:
                # scatter to specified GPU
                data = mmcv.parallel.scatter(data, [self.device])[0]
            else:
                for m in self.model.modules():
                    assert not isinstance(m, mmcv.runner.fp16_utils.Fp16Module), \
                        'Please disable fp16 when using CPU.'
                data['img_metas'] = [i.data[0] for i in data['img_metas']]
                data['img'] = [i.data[0] for i in data['img']]
            
            # 前向推理
            result = self.model(return_loss=False, rescale=True, **data)
            
        return result
    
    def process_directory(self, input_dir, output_dir, image_extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp']):
        """
        处理目录下的所有图像，保存彩色分割结果
        
        Args:
            input_dir (str): 输入目录
            output_dir (str): 输出目录
            image_extensions (list): 支持的图像扩展名
        """
        # 检查输入目录
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # 处理每张图像
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            
            try:
                # 读取图像
                img = mmcv.imread(image_path)
                
                # 进行推理 - 使用修复后的方法
                try:
                    result = inference_segmentor(self.model, img)
                except AttributeError:
                    # 如果inference_segmentor仍然有问题，使用自定义推理方法
                    result = self.inference_single_image(img)
                
                # 创建彩色掩码
                colored_mask = self.create_colored_mask(result[0])
                
                # 保存结果
                img_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f'{img_name}.jpg')
                success = cv2.imwrite(output_path, colored_mask)
                
                if success:
                    print(f"  Saved: {output_path}")
                else:
                    print(f"  Failed to save: {output_path}")
                    
            except Exception as e:
                print(f"  Error processing {image_path}: {str(e)}")
                continue
        
        print(f"\nProcessing completed! Results saved to: {output_dir}")


# 使用示例
if __name__ == '__main__':
    # 设置您的调色板
    palette = [
        [0, 0, 0],
        [64, 0, 128],
        [64, 64, 0],
        [0, 128, 192],
        [0, 0, 192],
        [128, 128, 0],
        [64, 64, 128],
        [192, 128, 128],
        [192, 64, 0]]
    
    # 创建推理对象
    segmentor = SegmentationInference(
        config_path='/public/home/wangshuo/ir_pretrain/UnIV_v1/SEG/MCMAE_SEG/configs/convmae/upernet_msrs.py',      # 替换为您的配置文件路径
        checkpoint_path='/public/home/wangshuo/NIPS2025/msrs_seg/iter_48000.pth',  # 替换为您的模型权重路径
        palette=palette,
        device='cpu'  # 或 'cuda:0'
    )
    
    # 处理目录下的所有图像
    segmentor.process_directory(
        input_dir='/public/home/wangshuo/NIPS2025/Vis/orgin',    # 替换为输入目录
        output_dir='/public/home/wangshuo/NIPS2025/Vis/seg'   # 替换为输出目录
    )