# -*- coding: utf-8 -*-
"""
主训练入口模块 (main.py)

功能说明:
    项目的主要训练入口，整合所有组件进行端到端训练

主要内容:
    - set_random_seed: 设置随机种子确保可重现性
    - check_environment: 检查运行环境 (Python版本、PyTorch、CUDA等)
    - validate_setup: 验证项目设置 (数据路径、模型路径、输出目录等)
    - main: 主训练函数
        * 环境验证
        * 配置加载和验证
        * 数据加载器创建
        * 模型创建和初始化
        * 训练器创建和执行
        * 结果汇总和保存

执行流程:
    1. 环境检查 -> 2. 配置验证 -> 3. 数据准备 -> 4. 模型创建
    5. 训练执行 -> 6. 结果保存 -> 7. 性能报告

支持功能:
    - 命令行参数解析
    - 多种设备支持 (CPU/CUDA)
    - 完整错误处理
    - 详细日志输出
    - WandB实验追踪

使用方法:
    # 直接运行
    python main.py
    
    # 通过run.py调用
    python run.py --mode train

输出:
    - 训练日志
    - 模型检查点
    - 最佳模型
    - 训练统计

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import sys
import torch
import random
import numpy as np
import argparse
from typing import Optional

# 导入自定义模块
from config import get_config, update_config
from data_loader import create_dataloaders, validate_dataset_paths
from models import create_feature_extractor
from trainer import Trainer


def set_random_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 确保可重现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"设置随机种子: {seed}")


def check_environment():
    """检查运行环境"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # PyTorch信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {'✅' if torch.cuda.is_available() else '❌'}")
    
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    
    print()


def validate_setup(config) -> bool:
    """验证设置"""
    print("=" * 50)
    print("设置验证")
    print("=" * 50)
    
    # 验证数据路径
    if not validate_dataset_paths(config.paths.ir_data_path, config.paths.vis_data_path):
        return False
    
    # 验证预训练模型
    if not os.path.exists(config.paths.pretrained_model_path):
        print(f"❌ 预训练模型不存在: {config.paths.pretrained_model_path}")
        return False
    
    print(f"✅ 预训练模型: {config.paths.pretrained_model_path}")
    
    # 创建输出目录
    config.create_directories()
    print(f"✅ 输出目录: {config.paths.output_dir}")
    
    return True


def main(args: Optional[argparse.Namespace] = None):
    """主函数"""
    print("🚀 SwinFuse特征提取器微调")
    print("重构版本 - 模块化、清晰、规范")
    print()
    
    # 加载配置
    config = get_config()
    
    # 如果有命令行参数，更新配置
    if args:
        if args.epochs:
            update_config(num_epochs=args.epochs)
        if args.batch_size:
            update_config(batch_size=args.batch_size)
        if args.lr:
            update_config(learning_rate=args.lr)
        if args.temperature:
            update_config(temperature=args.temperature)
    
    # 打印配置
    config.print_summary()
    
    # 环境检查
    check_environment()
    
    # 验证设置
    if not validate_setup(config):
        print("❌ 设置验证失败，退出程序")
        return False
    
    # 设置随机种子
    set_random_seed(config.random_seed)
    
    # 创建数据加载器
    print("正在创建数据加载器...")
    try:
        train_loader, val_loader = create_dataloaders(config)
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        return False
    
    # 创建模型
    print("正在创建特征提取器...")
    try:
        # 在多GPU模式下，不立即移动到设备，让trainer处理
        move_to_device = not config.training.use_multi_gpu
        model = create_feature_extractor(config, config.device, move_to_device=move_to_device)
        if model is None:
            print("❌ 创建特征提取器失败")
            return False
    except Exception as e:
        print(f"❌ 创建特征提取器失败: {e}")
        return False
    
    # 创建训练器
    print("正在初始化训练器...")
    try:
        trainer = Trainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            config=config,
            use_wandb=args.use_wandb if hasattr(args, 'use_wandb') else False
        )
    except Exception as e:
        print(f"❌ 初始化训练器失败: {e}")
        return False
    
    # 开始训练
    try:
        best_val_loss = trainer.train()
        print(f"\n✅ 训练成功完成")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"模型保存位置: {config.paths.checkpoint_dir}")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SwinFuse特征提取器微调训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 训练参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批量大小')
    parser.add_argument('--lr', type=float, help='学习率')
    
    # 损失参数
    parser.add_argument('--temperature', type=float, help='InfoNCE温度参数')
    parser.add_argument('--lambda-coral', type=float, help='CORAL损失权重')
    parser.add_argument('--lambda-barlow', type=float, help='Barlow损失权重')
    
    # 其他参数
    parser.add_argument('--config-only', action='store_true', help='仅显示配置不训练')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境不训练')
    
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    if args.config_only:
        # 仅显示配置
        config = get_config()
        config.print_summary()
        
    elif args.check_only:
        # 仅检查环境
        check_environment()
        config = get_config()
        validate_setup(config)
        
    else:
        # 开始训练
        success = main(args)
        sys.exit(0 if success else 1)
