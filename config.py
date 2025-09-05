# -*- coding: utf-8 -*-
"""
配置管理模块 (config.py)

功能说明:
    统一管理项目所有配置参数，采用dataclass结构化组织配置项

主要内容:
    - TrainingConfig: 训练相关配置 (学习率、批次大小、轮数等)
    - DataConfig: 数据相关配置 (图像尺寸、数据增强、归一化等)
    - ModelConfig: 模型相关配置 (投影头维度、SwinFuse参数等)
    - LossConfig: 损失函数配置 (InfoNCE、CORAL、Barlow权重等)
    - PathConfig: 路径配置 (数据路径、模型路径、输出路径等)
    - Config: 主配置类，整合所有子配置

使用方法:
    from config import get_config
    config = get_config()
    print(config.training.learning_rate)

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
    
    # 多GPU配置
    # use_multi_gpu: bool = False
    use_multi_gpu: bool = True
    gpu_ids: str = "0,1"  # 使用的GPU编号，逗号分隔，如"0,1,2,3"
    distributed: bool = False  # 是否使用分布式训练(DDP)
    local_rank: int = -1  # 分布式训练的本地rank


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
        
        # GPU相关配置
        self.num_gpus = torch.cuda.device_count()
        self.available_gpus = list(range(self.num_gpus)) if torch.cuda.is_available() else []
        
        # 日志配置
        self.print_interval = 10
        self.save_interval = 5
        self.eval_interval = 1
    
    def setup_gpu_config(self, gpu_ids: str = None, use_multi_gpu: bool = False, distributed: bool = False):
        """设置GPU配置"""
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将使用CPU")
            self.device = "cpu"
            self.training.use_multi_gpu = False
            self.training.distributed = False
            return
        
        # 解析GPU ID
        if gpu_ids:
            try:
                requested_gpus = [int(x.strip()) for x in gpu_ids.split(',')]
                # 验证GPU ID有效性
                valid_gpus = [gpu_id for gpu_id in requested_gpus if gpu_id < self.num_gpus]
                if not valid_gpus:
                    print(f"❌ 指定的GPU不可用，可用GPU: {self.available_gpus}")
                    valid_gpus = [0] if self.available_gpus else []
                elif len(valid_gpus) != len(requested_gpus):
                    invalid_gpus = set(requested_gpus) - set(valid_gpus)
                    print(f"⚠️ GPU {invalid_gpus} 不可用，将使用: {valid_gpus}")
                
                self.available_gpus = valid_gpus
                self.training.gpu_ids = ','.join(map(str, valid_gpus))
            except ValueError:
                print(f"❌ 无效的GPU ID格式: {gpu_ids}")
                self.available_gpus = [0] if self.num_gpus > 0 else []
        
        # 设置多GPU配置
        if use_multi_gpu and len(self.available_gpus) > 1:
            self.training.use_multi_gpu = True
            self.training.distributed = distributed
            self.device = f"cuda:{self.available_gpus[0]}"
            print(f"✅ 多GPU模式: 使用GPU {self.available_gpus}")
            
            if distributed:
                print("✅ 分布式训练模式已启用")
        elif self.available_gpus:
            self.training.use_multi_gpu = False
            self.training.distributed = False
            self.device = f"cuda:{self.available_gpus[0]}"
            print(f"✅ 单GPU模式: 使用GPU {self.available_gpus[0]}")
        else:
            self.device = "cpu"
            print("⚠️ 无可用GPU，使用CPU")
    
    def get_effective_batch_size(self):
        """获取有效批次大小"""
        base_batch_size = self.training.batch_size
        if self.training.use_multi_gpu:
            # 多GPU时，每个GPU处理的批次大小
            num_gpus = len(self.available_gpus)
            return base_batch_size * num_gpus
        return base_batch_size
    
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
        
        print(f"\n设备配置:")
        print(f"  主设备: {self.device}")
        print(f"  可用GPU数量: {self.num_gpus}")
        print(f"  使用的GPU: {self.available_gpus}")
        print(f"  多GPU模式: {self.training.use_multi_gpu}")
        if self.training.use_multi_gpu:
            print(f"  分布式训练: {self.training.distributed}")
            print(f"  有效批次大小: {self.get_effective_batch_size()}")
        
        print(f"\n训练配置:")
        print(f"  轮数: {self.training.num_epochs}")
        print(f"  批量: {self.training.batch_size}")
        print(f"  学习率: {self.training.learning_rate}")
        
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
