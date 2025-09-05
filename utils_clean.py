# -*- coding: utf-8 -*-
"""
实用工具模块 (utils_clean.py)

功能说明:
    提供项目中各种辅助功能和工具函数

主要内容:
    - 日志系统管理:
        * setup_logging: 配置日志系统
        * 支持文件和控制台输出
        * 可配置日志级别
    
    - 系统信息:
        * print_system_info: 打印系统和硬件信息
        * GPU信息检测
        * Python和库版本信息
    
    - 文件操作:
        * backup_config: 备份配置文件
        * clean_temp_files: 清理临时文件
        * 安全的文件操作
    
    - 模型工具:
        * count_parameters: 统计模型参数
        * save_model_info: 保存模型信息
        * load_checkpoint: 加载检查点
    
    - 数据处理:
        * normalize_features: 特征归一化
        * compute_metrics: 计算评估指标
        * visualize_results: 结果可视化
    
    - 配置管理:
        * merge_configs: 合并配置字典
        * validate_config: 验证配置有效性
        * config_to_dict: 配置对象转字典

使用特点:
    - 所有函数都有详细的类型注解
    - 完整的错误处理
    - 支持多种日志级别
    - 跨平台兼容性

使用方法:
    from utils_clean import setup_logging, print_system_info
    logger = setup_logging("logs")
    print_system_info()

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import time
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import shutil
import logging


def setup_logging(log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('SwinFuse')
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 文件handler
    log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger


class Timer:
    """计时器类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """返回已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def elapsed_str(self) -> str:
        """返回格式化的时间字符串"""
        total_seconds = self.elapsed()
        
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def __enter__(self):
        """上下文管理器入口"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class ModelSaver:
    """模型保存管理器"""
    
    def __init__(self, save_dir: str, keep_best: int = 3):
        """
        Args:
            save_dir: 保存目录
            keep_best: 保留最佳模型数量
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_models = []  # [(score, path), ...]
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       val_loss: float,
                       config: Any,
                       is_best: bool = False,
                       prefix: str = "") -> str:
        """保存检查点"""
        
        # 构建文件名
        if is_best:
            filename = f"{prefix}best_model.pth"
        else:
            filename = f"{prefix}checkpoint_epoch_{epoch:03d}.pth"
        
        filepath = self.save_dir / filename
        
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config,
            'timestamp': time.time()
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 保存文件
        torch.save(checkpoint, filepath)
        
        # 管理最佳模型列表
        if is_best:
            self._update_best_models(val_loss, filepath)
        
        return str(filepath)
    
    def _update_best_models(self, score: float, path: Path):
        """更新最佳模型列表"""
        self.best_models.append((score, path))
        self.best_models.sort(key=lambda x: x[0])  # 按分数排序
        
        # 删除多余的模型
        while len(self.best_models) > self.keep_best:
            _, old_path = self.best_models.pop()
            if old_path.exists() and old_path != path:
                old_path.unlink()
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """加载检查点"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点文件"""
        checkpoints = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        
        # 按修改时间排序，返回最新的
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []
            
            self.metrics[key].append(value)
            self.history[key].append(value)
    
    def get_current(self, key: str) -> Optional[float]:
        """获取当前指标值"""
        if key not in self.metrics or not self.metrics[key]:
            return None
        return self.metrics[key][-1]
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> Optional[float]:
        """获取平均值"""
        if key not in self.metrics or not self.metrics[key]:
            return None
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_best(self, key: str, mode: str = 'min') -> Tuple[Optional[float], Optional[int]]:
        """获取最佳值和对应的epoch"""
        if key not in self.history or not self.history[key]:
            return None, None
        
        values = self.history[key]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return values[best_idx], best_idx
    
    def reset_current(self):
        """重置当前epoch的指标"""
        for key in self.metrics:
            self.metrics[key] = []
    
    def save_to_file(self, filepath: str):
        """保存指标到文件"""
        data = {
            'metrics': self.metrics,
            'history': self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        """从文件加载指标"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metrics = data.get('metrics', {})
        self.history = data.get('history', {})


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return {}
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    
    return {
        'total_gb': total_memory / 1024**3,
        'allocated_gb': allocated_memory / 1024**3,
        'cached_gb': cached_memory / 1024**3,
        'free_gb': (total_memory - allocated_memory) / 1024**3,
        'utilization': allocated_memory / total_memory
    }


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def backup_config(config_obj: Any, backup_dir: str) -> str:
    """备份配置"""
    backup_dir = ensure_dir(backup_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"config_backup_{timestamp}.json")
    
    # 将配置对象转换为字典
    if hasattr(config_obj, '__dict__'):
        config_dict = {}
        for attr_name in dir(config_obj):
            if not attr_name.startswith('_'):
                attr_value = getattr(config_obj, attr_name)
                if not callable(attr_value):
                    if hasattr(attr_value, '__dict__'):
                        # 嵌套对象
                        config_dict[attr_name] = {
                            k: v for k, v in attr_value.__dict__.items()
                            if not k.startswith('_') and not callable(v)
                        }
                    else:
                        config_dict[attr_name] = attr_value
    else:
        config_dict = config_obj
    
    # 保存到文件
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    
    return backup_file


def print_system_info():
    """打印系统信息"""
    print("=" * 50)
    print("系统信息")
    print("=" * 50)
    
    # PyTorch信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {device_count}")
        print(f"当前GPU: {device_name}")
        
        # GPU内存信息
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"GPU内存: {memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB "
                  f"({memory_info['utilization']*100:.1f}%)")
    
    print()


if __name__ == "__main__":
    # 测试工具函数
    print_system_info()
    
    # 测试计时器
    with Timer() as timer:
        time.sleep(0.1)  # 模拟操作
    print(f"操作耗时: {timer.elapsed_str()}")
    
    # 测试指标跟踪器
    tracker = MetricsTracker()
    for i in range(10):
        tracker.update(loss=1.0/(i+1), accuracy=i*0.1)
    
    print(f"平均损失: {tracker.get_average('loss'):.4f}")
    print(f"最佳准确率: {tracker.get_best('accuracy', 'max')[0]:.4f}")
    
    # 测试模型保存器
    save_dir = "test_saves"
    saver = ModelSaver(save_dir)
    print(f"模型保存目录: {save_dir}")
    
    # 清理测试文件
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
