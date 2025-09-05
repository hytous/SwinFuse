#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SwinFuse特征提取器微调项目启动脚本
支持训练、测试、配置验证等功能

使用示例:
    python run.py --mode train                    # 开始训练
    python run.py --mode test --checkpoint best   # 测试最佳模型
    python run.py --mode validate                 # 验证配置和环境
    python run.py --mode clean                    # 清理临时文件

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils_clean import setup_logging, print_system_info, backup_config
from main import main as training_main


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SwinFuse特征提取器微调项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --mode train                     # 开始训练
  %(prog)s --mode train --resume            # 从检查点恢复训练
  %(prog)s --mode test --checkpoint best    # 测试最佳模型
  %(prog)s --mode validate                  # 验证配置和环境
  %(prog)s --mode clean                     # 清理临时文件
        """
    )
    
    # 主要模式
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'validate', 'clean'],
        default='train',
        help='运行模式 (默认: train)'
    )
    
    # 训练相关参数
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (可选，使用默认配置)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从最新检查点恢复训练'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best',
        help='检查点文件路径或类型 (best/latest/路径)'
    )
    
    # 数据路径
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据集根目录'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='预训练模型路径'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率'
    )
    
    # 其他选项
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU设备ID'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='数据加载工作进程数'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不备份配置文件'
    )
    
    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace):
    """根据命令行参数更新配置"""
    
    # 数据配置
    if args.data_dir:
        config.data.data_root = args.data_dir
    
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    if args.workers:
        config.data.num_workers = args.workers
    
    # 训练配置
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    # 模型配置
    if args.pretrained:
        config.model.pretrained_path = args.pretrained
    
    # 路径配置
    if args.gpu is not None:
        config.paths.device = f"cuda:{args.gpu}"


def validate_environment():
    """验证运行环境"""
    print("🔍 验证运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Python版本过低，需要Python 3.7+")
        return False
    
    # 检查必要的包
    required_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python',
        'matplotlib', 'seaborn', 'tqdm', 'argparse'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
        print("请使用 pip install 安装缺少的包")
        return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")
        return False
    
    print("✅ 环境验证通过")
    return True


def validate_config(config: Config):
    """验证配置"""
    print("🔍 验证配置...")
    
    errors = []
    warnings = []
    
    # 检查必要路径
    if not config.data.data_root:
        errors.append("未设置数据集根目录")
    elif not os.path.exists(config.data.data_root):
        warnings.append(f"数据集目录不存在: {config.data.data_root}")
    
    if config.model.pretrained_path and not os.path.exists(config.model.pretrained_path):
        warnings.append(f"预训练模型文件不存在: {config.model.pretrained_path}")
    
    # 检查参数合理性
    if config.training.learning_rate <= 0:
        errors.append("学习率必须大于0")
    
    if config.data.batch_size <= 0:
        errors.append("批次大小必须大于0")
    
    if config.training.num_epochs <= 0:
        errors.append("训练轮数必须大于0")
    
    # 输出结果
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("⚠️  配置警告:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("✅ 配置验证通过")
    return True


def clean_temporary_files():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    
    temp_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.pytest_cache',
        '.coverage',
        'logs/*.tmp',
        'checkpoints/*.tmp'
    ]
    
    project_root = Path(__file__).parent
    cleaned_count = 0
    
    # 清理 __pycache__ 目录
    for pycache in project_root.rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            cleaned_count += 1
            print(f"   删除: {pycache}")
    
    # 清理 .pyc 文件
    for pyc_file in project_root.rglob('*.pyc'):
        pyc_file.unlink()
        cleaned_count += 1
    
    # 清理 .pyo 文件
    for pyo_file in project_root.rglob('*.pyo'):
        pyo_file.unlink()
        cleaned_count += 1
    
    print(f"✅ 清理完成，删除了 {cleaned_count} 个临时文件/目录")


def mode_train(args):
    """训练模式"""
    print("🚀 开始训练模式")
    
    # 验证环境
    if not validate_environment():
        return 1
    
    # 加载配置
    config = Config()
    
    # 应用命令行参数
    update_config_from_args(config, args)
    
    # 验证配置
    if not validate_config(config):
        return 1
    
    # 备份配置
    if not args.no_backup:
        backup_path = backup_config(config, config.paths.output_dir)
        print(f"📝 配置已备份到: {backup_path}")
    
    # 设置日志
    logger = setup_logging(
        config.paths.output_dir,
        getattr(__import__('logging'), args.log_level)
    )
    
    # 打印系统信息
    print_system_info()
    
    # 开始训练
    try:
        # 设置参数
        sys.argv = ['main.py']
        if args.resume:
            sys.argv.append('--resume')
        
        # 调用主训练函数
        training_main()
        print("✅ 训练完成")
        return 0
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        logger.error(f"训练失败: {e}", exc_info=True)
        return 1


def mode_test(args):
    """测试模式"""
    print("🧪 开始测试模式")
    
    # 验证环境
    if not validate_environment():
        return 1
    
    config = Config()
    update_config_from_args(config, args)
    
    # 查找检查点文件
    checkpoint_path = None
    
    if args.checkpoint == 'best':
        checkpoint_path = os.path.join(config.paths.checkpoint_dir, 'best_model.pth')
    elif args.checkpoint == 'latest':
        # 查找最新的检查点
        checkpoint_dir = Path(config.paths.checkpoint_dir)
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            if checkpoints:
                checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    else:
        checkpoint_path = args.checkpoint
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return 1
    
    print(f"📁 使用检查点: {checkpoint_path}")
    
    # TODO: 实现测试逻辑
    print("⚠️  测试功能待实现")
    return 0


def mode_validate(args):
    """验证模式"""
    print("✅ 开始验证模式")
    
    # 验证环境
    env_ok = validate_environment()
    
    # 验证配置
    config = Config()
    update_config_from_args(config, args)
    config_ok = validate_config(config)
    
    # 打印总结
    print("\n" + "="*50)
    print("验证总结")
    print("="*50)
    print(f"环境检查: {'✅ 通过' if env_ok else '❌ 失败'}")
    print(f"配置检查: {'✅ 通过' if config_ok else '❌ 失败'}")
    
    if env_ok and config_ok:
        print("\n🎉 所有检查都通过，可以开始训练！")
        return 0
    else:
        print("\n❌ 存在问题，请解决后再运行")
        return 1


def mode_clean(args):
    """清理模式"""
    clean_temporary_files()
    return 0


def main():
    """主函数"""
    args = parse_arguments()
    
    # 打印启动信息
    print("="*60)
    print("🔥 SwinFuse特征提取器微调项目")
    print("="*60)
    print(f"运行模式: {args.mode}")
    print(f"项目目录: {os.path.dirname(os.path.abspath(__file__))}")
    print()
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        return mode_train(args)
    elif args.mode == 'test':
        return mode_test(args)
    elif args.mode == 'validate':
        return mode_validate(args)
    elif args.mode == 'clean':
        return mode_clean(args)
    else:
        print(f"❌ 未知模式: {args.mode}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
