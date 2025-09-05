# -*- coding: utf-8 -*-
"""
数据加载模块 (data_loader.py)

功能说明:
    处理RoadScene数据集的加载、预处理和增强，为训练提供IR-可见光图像对

主要内容:
    - RoadSceneDataset: 自定义数据集类
        * 加载红外(IR)和可见光(VIS)图像对
        * 图像预处理 (resize、归一化、转tensor)
        * 数据增强 (随机翻转、旋转等)
    - create_dataloaders: 创建训练和验证数据加载器
    - validate_dataset_paths: 验证数据集路径有效性

数据格式要求:
    - IR和VIS图像必须有相同的文件名
    - 支持常见图像格式 (jpg, png, bmp等)
    - 图像会被resize到指定尺寸 (默认224x224)

使用方法:
    from data_loader import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from typing import Tuple, List, Optional


class RoadSceneDataset(Dataset):
    """RoadScene数据集类"""
    
    def __init__(self, 
                 ir_path: str, 
                 vis_path: str, 
                 image_size: int = 224,
                 augment: bool = False,
                 normalize_mean: float = 0.5,
                 normalize_std: float = 0.5):
        """
        Args:
            ir_path: 红外图像文件夹路径
            vis_path: 可见光图像文件夹路径
            image_size: 目标图像尺寸
            augment: 是否使用数据增强
            normalize_mean: 归一化均值
            normalize_std: 归一化标准差
        """
        self.ir_path = ir_path
        self.vis_path = vis_path
        self.image_size = image_size
        self.augment = augment
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # 获取匹配的文件对
        self.file_pairs = self._get_matched_pairs()
        print(f"数据集加载完成: 共{len(self.file_pairs)}对图像")
    
    def _get_matched_pairs(self) -> List[Tuple[str, str]]:
        """获取匹配的红外-可见光图像对"""
        # 获取文件列表
        ir_files = self._get_image_files(self.ir_path)
        vis_files = self._get_image_files(self.vis_path)
        
        # 提取文件名（不含扩展名）
        ir_names = {os.path.splitext(f)[0]: f for f in ir_files}
        vis_names = {os.path.splitext(f)[0]: f for f in vis_files}
        
        # 找到匹配的文件对
        matched_pairs = []
        for name in ir_names:
            if name in vis_names:
                matched_pairs.append((ir_names[name], vis_names[name]))
        
        if len(matched_pairs) == 0:
            raise ValueError("没有找到匹配的图像对")
        
        return matched_pairs
    
    def _get_image_files(self, path: str) -> List[str]:
        """获取路径下的所有图像文件"""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in os.listdir(path) 
                if f.lower().endswith(extensions)]
        return sorted(files)
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """获取一对图像"""
        ir_file, vis_file = self.file_pairs[idx]
        
        # 读取图像
        ir_img = self._load_image(os.path.join(self.ir_path, ir_file))
        vis_img = self._load_image(os.path.join(self.vis_path, vis_file))
        
        # 数据增强
        if self.augment:
            ir_img, vis_img = self._apply_augmentation(ir_img, vis_img)
        
        # 转换为张量
        ir_tensor = self._to_tensor(ir_img)
        vis_tensor = self._to_tensor(vis_img)
        
        return ir_tensor, vis_tensor, ir_file
    
    def _load_image(self, path: str) -> Image.Image:
        """加载并预处理图像"""
        img = Image.open(path).convert('L')  # 转为灰度图
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return img
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL图像转换为张量"""
        # 转换为numpy数组
        array = np.array(img, dtype=np.float32) / 255.0
        
        # 归一化
        array = (array - self.normalize_mean) / self.normalize_std
        
        # 转换为张量并添加通道维度
        tensor = torch.from_numpy(array).unsqueeze(0)  # [1, H, W]
        return tensor
    
    def _apply_augmentation(self, ir_img: Image.Image, vis_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """应用数据增强（保持两个图像的一致性）"""
        # 随机水平翻转
        if random.random() > 0.5:
            ir_img = ir_img.transpose(Image.FLIP_LEFT_RIGHT)
            vis_img = vis_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            ir_img = ir_img.transpose(Image.FLIP_TOP_BOTTOM)
            vis_img = vis_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 小幅旋转
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            ir_img = ir_img.rotate(angle, fillcolor=128)
            vis_img = vis_img.rotate(angle, fillcolor=128)
        
        return ir_img, vis_img


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    # 创建完整数据集
    full_dataset = RoadSceneDataset(
        ir_path=config.paths.ir_data_path,
        vis_path=config.paths.vis_data_path,
        image_size=config.data.image_size,
        augment=config.data.use_augmentation,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std
    )
    
    # 分割数据集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * config.data.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    
    return train_loader, val_loader


def validate_dataset_paths(ir_path: str, vis_path: str) -> bool:
    """验证数据集路径"""
    if not os.path.exists(ir_path):
        print(f"❌ 红外图像路径不存在: {ir_path}")
        return False
    
    if not os.path.exists(vis_path):
        print(f"❌ 可见光图像路径不存在: {vis_path}")
        return False
    
    # 检查图像数量
    try:
        dataset = RoadSceneDataset(ir_path, vis_path)
        if len(dataset) == 0:
            print("❌ 没有找到匹配的图像对")
            return False
        print(f"✅ 找到 {len(dataset)} 对匹配图像")
        return True
    except Exception as e:
        print(f"❌ 数据集验证失败: {e}")
        return False


if __name__ == "__main__":
    # 测试数据加载
    from config import get_config
    
    config = get_config()
    
    # 验证路径
    if validate_dataset_paths(config.paths.ir_data_path, config.paths.vis_data_path):
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(config)
        
        # 测试加载一个批次
        for ir_batch, vis_batch, filenames in train_loader:
            print(f"批次形状: IR={ir_batch.shape}, VIS={vis_batch.shape}")
            print(f"文件名示例: {filenames[0]}")
            break
