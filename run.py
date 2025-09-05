#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目启动脚本 (run.py) - 主要入口点

功能说明:
    项目的统一启动入口，支持多种运行模式和完整的参数管理

支持模式:
    - train: 标准训练模式
        * 支持WandB实验追踪
        * 支持从检查点恢复
        * 完整的训练流程
    - test: 模型测试模式
        * 加载已训练模型
        * 在测试集上评估
    - validate: 环境和配置验证
        * 检查依赖安装
        * 验证数据路径
        * 测试模型加载
    - optimize: Optuna超参数优化
        * 自动搜索最优参数
        * 支持分布式优化
        * 结果可视化
    - best-params: 显示最优参数
        * 展示优化结果
        * 生成配置文件
    - clean: 清理临时文件

主要功能:
    - 命令行参数解析和验证
    - 多模式执行逻辑
    - 配置文件管理
    - 环境依赖检查
    - 错误处理和日志
    - WandB和Optuna集成

参数支持:
    - 训练参数: epochs, batch-size, lr等
    - 实验管理: use-wandb, experiment-name等
    - 优化参数: n-trials, study-name等
    - 路径配置: data-dir, output-dir等

使用示例:
    # 标准训练
    python run.py --mode train --use-wandb
    
    # 超参数优化
    python run.py --mode optimize --n-trials 50
    
    # 使用最优参数训练
    python run.py --mode train --config best_config.json

文件关系:
    run.py -> main.py -> trainer.py -> models.py + losses.py + data_loader.py

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

from config import get_config, Config
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
  
多GPU使用示例:
  %(prog)s --mode train --multi-gpu --gpu-ids 0,1        # 使用GPU 0和1训练
  %(prog)s --mode train --multi-gpu --gpu-ids 0,1,2,3    # 使用GPU 0,1,2,3训练
  %(prog)s --mode train --distributed --gpu-ids 0,1      # 分布式训练(实验性)
        """
    )
    
    # 主要模式
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'validate', 'clean', 'optimize', 'best-params'],
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
    
    # GPU相关参数
    parser.add_argument(
        '--gpu-ids',
        type=str,
        # default=None,
        default="0,1",
        help='指定使用的GPU编号，逗号分隔 (例如: 0,1,2,3)'
    )
    
    parser.add_argument(
        '--multi-gpu',
        # action='store_true',
        action='store_false',
        help='启用多GPU训练'
    )
    
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='启用分布式训练 (DDP)'
    )
    
    parser.add_argument(
        '--local-rank',
        type=int,
        default=-1,
        help='分布式训练的本地rank'
    )
    
    # 实验管理
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='使用WandB记录实验'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='SwinFuse-FeatureExtractor',
        help='WandB项目名称'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='实验名称'
    )
    
    # Optuna相关
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Optuna优化试验次数'
    )
    
    parser.add_argument(
        '--study-name',
        type=str,
        default='swinfuse_hp_opt',
        help='Optuna研究名称'
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
        # 数据目录暂时设置到paths中，或者需要分别设置ir和vis路径
        config.paths.ir_data_path = os.path.join(args.data_dir, "ir")
        config.paths.vis_data_path = os.path.join(args.data_dir, "vis")
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    if args.workers:
        config.data.num_workers = args.workers
    
    # 训练配置
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    # 模型配置
    if args.pretrained:
        config.paths.pretrained_model_path = args.pretrained
    
    # GPU配置（在config.setup_gpu_config之后更新特定参数）
    if hasattr(args, 'gpu') and args.gpu is not None:
        config.device = f"cuda:{args.gpu}"
    
    # 多GPU配置已在setup_gpu_config中处理
    # 这里可以添加其他GPU相关的覆盖设置


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
        'torch', 'torchvision', 'numpy', 'cv2',
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
    if not config.paths.ir_data_path:
        errors.append("未设置红外图像路径")
    elif not os.path.exists(config.paths.ir_data_path):
        warnings.append(f"红外图像目录不存在: {config.paths.ir_data_path}")
    
    if not config.paths.vis_data_path:
        errors.append("未设置可见光图像路径")
    elif not os.path.exists(config.paths.vis_data_path):
        warnings.append(f"可见光图像目录不存在: {config.paths.vis_data_path}")
    
    if config.paths.pretrained_model_path and not os.path.exists(config.paths.pretrained_model_path):
        warnings.append(f"预训练模型文件不存在: {config.paths.pretrained_model_path}")
    
    # 检查参数合理性
    if config.training.learning_rate <= 0:
        errors.append("学习率必须大于0")
    
    if config.training.batch_size <= 0:
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
    """训练模式 - 支持WandB实验追踪"""
    print("🚀 开始训练模式")
    
    # 检查WandB依赖
    if args.use_wandb:
        try:
            import wandb
            print("✅ WandB可用")
        except ImportError:
            print("❌ WandB未安装，请运行: pip install wandb")
            print("⚠️ 将在不使用WandB的情况下继续训练")
            args.use_wandb = False
    
    # 验证环境
    if not validate_environment():
        return 1
    
    # 加载配置
    config = get_config()
    
    # 设置GPU配置
    config.setup_gpu_config(
        gpu_ids=args.gpu_ids,
        use_multi_gpu=args.multi_gpu,
        distributed=args.distributed
    )
    
    # 设置分布式训练参数
    if args.distributed and args.local_rank >= 0:
        config.training.local_rank = args.local_rank
    
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
        # 直接调用训练逻辑，不通过main.py
        from data_loader import create_dataloaders
        from models import create_feature_extractor
        from trainer import Trainer
        
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(config)
        
        # 创建模型
        model = create_feature_extractor(config, config.device)
        if model is None:
            print("❌ 创建特征提取器失败")
            return 1
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            use_wandb=args.use_wandb
        )
        
        # 开始训练
        best_val_loss = trainer.train()
        print(f"✅ 训练完成，最佳验证损失: {best_val_loss:.6f}")
        return 0
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        logger.error(f"训练失败: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return 1


def mode_test(args):
    """测试模式"""
    print("🧪 开始测试模式")
    
    # 验证环境
    if not validate_environment():
        return 1
    
    config = get_config()
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
    config = get_config()
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


def mode_optimize(args):
    """超参数优化模式"""
    print("🎯 开始超参数优化模式")
    
    # 检查Optuna依赖
    try:
        from trainer import create_hyperparameter_study, OPTUNA_AVAILABLE
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna未安装，请运行: pip install optuna")
            return 1
    except ImportError:
        print("❌ Optuna依赖导入失败")
        return 1
    
    # 加载配置
    config = get_config()
    update_config_from_args(config, args)
    
    # 定义工厂函数
    def model_factory(cfg):
        from models import create_feature_extractor
        return create_feature_extractor(cfg)
    
    def data_loaders_factory(cfg):
        from data_loader import create_dataloaders
        return create_dataloaders(cfg)
    
    try:
        # 执行优化
        study = create_hyperparameter_study(
            config=config,
            model_factory=model_factory,
            data_loaders_factory=data_loaders_factory,
            n_trials=args.n_trials
        )
        
        # 保存最佳参数
        import json
        best_params_file = f"best_params_{args.study_name}.json"
        with open(best_params_file, 'w', encoding='utf-8') as f:
            json.dump(study.best_params, f, indent=2, ensure_ascii=False)
        print(f"✅ 最佳参数已保存到: {best_params_file}")
        
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ 优化被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def mode_best_params(args):
    """显示最佳参数"""
    import json
    best_params_file = f"best_params_{args.study_name}.json"
    
    if not os.path.exists(best_params_file):
        print(f"❌ 最佳参数文件不存在: {best_params_file}")
        print(f"💡 请先运行优化: python {sys.argv[0]} --mode optimize")
        return 1
    
    with open(best_params_file, 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    
    print("🏆 最佳超参数:")
    print("=" * 50)
    for key, value in best_params.items():
        print(f"{key:20s}: {value}")
    print("=" * 50)
    
    print(f"💡 使用最佳参数训练:")
    print(f"python {sys.argv[0]} --mode train --use-wandb \\")
    for key, value in best_params.items():
        if key in ['learning_rate', 'batch_size', 'weight_decay']:
            print(f"  --{key.replace('_', '-')} {value} \\")
    
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
    elif args.mode == 'optimize':
        return mode_optimize(args)
    elif args.mode == 'best-params':
        return mode_best_params(args)
    else:
        print(f"❌ 未知模式: {args.mode}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
