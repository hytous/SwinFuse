# -*- coding: utf-8 -*-
"""
SwinFuse特征提取器微调项目
统一配置管理模块

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # 学习率调度
    scheduler_type: str = "cosine"  # "cosine", "step", "none"
    warmup_epochs: int = 3
    min_lr: float = 1e-7
    
    # 早停
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-6


@dataclass 
class DataConfig:
    """数据配置"""
    image_size: int = 224
    val_split: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True
    
    # 数据增强
    use_augmentation: bool = True
    random_flip: bool = True
    random_rotation: float = 5.0
    
    # 归一化
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


@dataclass
class ModelConfig:
    """模型配置"""
    # SwinFuse参数
    input_channels: int = 1
    embed_dim: int = 96
    window_size: int = 7
    
    # 投影头参数
    projection_input_dim: int = 96
    projection_hidden_dim: int = 256
    projection_output_dim: int = 128


@dataclass
class LossConfig:
    """损失函数配置"""
    loss_type: str = "combined"  # "combined", "infonce", "coral", "barlow"
    
    # InfoNCE参数
    temperature: float = 0.07
    
    # 组合损失权重
    lambda_coral: float = 0.02
    lambda_barlow: float = 0.01
    lambda_off_diag: float = 0.005


@dataclass
class PathConfig:
    """路径配置"""
    # 数据路径
    ir_data_path: str = "D:/wbh/Registration/data/图像融合常用数据集整理/红外和可见光图像融合数据集/RoadScene/RoadScene-master/ir"
    vis_data_path: str = "D:/wbh/Registration/data/图像融合常用数据集整理/红外和可见光图像融合数据集/RoadScene/RoadScene-master/vi"
    
    # 模型路径
    pretrained_model_path: str = "SwinFuse_model/Final_epoch_50_Mon_Feb_14_17_37_05_2022_1e3.model"
    
    # 输出路径
    output_dir: str = "finetune_results"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """后处理：创建完整路径"""
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")


class Config:
    """主配置类，整合所有配置"""
    
    def __init__(self):
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.loss = LossConfig()
        self.paths = PathConfig()
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.random_seed = 42
        
        # 日志配置
        self.print_interval = 10
        self.save_interval = 5
        self.eval_interval = 1
    
    def create_directories(self):
        """创建必要的目录"""
        dirs = [self.paths.output_dir, self.paths.checkpoint_dir, self.paths.log_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def validate_paths(self) -> bool:
        """验证路径有效性"""
        paths_to_check = [
            (self.paths.ir_data_path, "红外图像路径"),
            (self.paths.vis_data_path, "可见光图像路径"),
            (self.paths.pretrained_model_path, "预训练模型路径")
        ]
        
        all_valid = True
        for path, desc in paths_to_check:
            if not os.path.exists(path):
                print(f"❌ {desc}不存在: {path}")
                all_valid = False
            else:
                print(f"✅ {desc}: {path}")
        
        return all_valid
    
    def print_summary(self):
        """打印配置摘要"""
        print("=" * 60)
        print("SwinFuse微调配置摘要")
        print("=" * 60)
        
        print(f"\n训练配置:")
        print(f"  轮数: {self.training.num_epochs}")
        print(f"  批量: {self.training.batch_size}")
        print(f"  学习率: {self.training.learning_rate}")
        print(f"  设备: {self.device}")
        
        print(f"\n损失配置:")
        print(f"  类型: {self.loss.loss_type}")
        print(f"  温度: {self.loss.temperature}")
        print(f"  CORAL权重: {self.loss.lambda_coral}")
        print(f"  Barlow权重: {self.loss.lambda_barlow}")
        
        print(f"\n数据配置:")
        print(f"  图像尺寸: {self.data.image_size}")
        print(f"  验证比例: {self.data.val_split}")
        print(f"  数据增强: {self.data.use_augmentation}")
        
        print("=" * 60)


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置实例"""
    return config


def update_config(**kwargs):
    """更新配置参数"""
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.loss, key):
            setattr(config.loss, key, value)
        elif hasattr(config.paths, key):
            setattr(config.paths, key, value)
        else:
            setattr(config, key, value)


if __name__ == "__main__":
    # 测试配置
    config = get_config()
    config.print_summary()
    config.validate_paths()
