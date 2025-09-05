# -*- coding: utf-8 -*-
"""
训练器模块 (trainer.py)

功能说明:
    管理完整的训练流程，集成Optuna超参数优化和WandB实验追踪

主要内容:
    - EarlyStopping: 早停机制类
        * 监控验证损失变化
        * 避免过拟合
    - TrainingLogger: 训练日志记录器
        * 本地日志保存
        * WandB实验追踪集成
        * 实时指标记录
    - Trainer: 主训练器类
        * 完整训练循环管理
        * 优化器和调度器创建
        * 模型保存和加载
        * Optuna剪枝支持
        * WandB可视化集成
    - create_hyperparameter_study: Optuna超参数优化
        * 自动搜索最优超参数
        * 支持并行优化
        * 智能剪枝机制
    - create_trial_config: 为优化试验创建配置

训练流程:
    1. 数据加载 -> 2. 模型前向 -> 3. 损失计算 -> 4. 反向传播
    5. 参数更新 -> 6. 验证评估 -> 7. 早停检查 -> 8. 模型保存

集成功能:
    - WandB: 实时监控、实验对比、模型管理
    - Optuna: 超参数优化、试验剪枝、结果分析

使用方法:
    # 普通训练
    trainer = Trainer(model, train_loader, val_loader, config, use_wandb=True)
    trainer.train()
    
    # 超参数优化
    study = create_hyperparameter_study(config, model_factory, data_factory)

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional

from losses import create_loss_function, compute_feature_similarity, CombinedLoss

# 可选依赖导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB未安装，可视化功能将被禁用")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna未安装，超参数优化功能将被禁用")


class EarlyStopping:
    """早停类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """检查是否应该早停"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class TrainingLogger:
    """训练日志记录器 - 集成WandB支持"""
    
    def __init__(self, log_dir: str, use_wandb: bool = False, project_name: str = "SwinFuse-FeatureExtractor"):
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        os.makedirs(log_dir, exist_ok=True)
        
        # 记录列表
        self.train_losses = []
        self.val_losses = []
        self.loss_components = {'infonce': [], 'coral': [], 'barlow': []}
        self.similarities = []
        self.learning_rates = []
        
        # 初始化WandB
        if self.use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    name=f"swinfuse_{int(time.time())}",
                    tags=["feature-extraction", "contrastive-learning", "ir-visible"]
                )
                print("✅ WandB初始化成功")
            except Exception as e:
                print(f"⚠️ WandB初始化失败: {e}")
                self.use_wandb = False
        
    def log_config(self, config):
        """记录配置到WandB"""
        if self.use_wandb:
            wandb.config.update({
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.batch_size,
                "num_epochs": config.training.num_epochs,
                "weight_decay": config.training.weight_decay,
                "scheduler_type": config.training.scheduler_type,
                "projection_input_dim": config.model.projection_input_dim,
                "projection_hidden_dim": config.model.projection_hidden_dim,
                "projection_output_dim": config.model.projection_output_dim,
                "loss_type": config.loss.loss_type,
                "temperature": config.loss.temperature,
                "lambda_coral": config.loss.lambda_coral,
                "lambda_barlow": config.loss.lambda_barlow,
                "image_size": config.data.image_size,
                "use_augmentation": config.data.use_augmentation,
                "device": config.device
            })
    
    def log_epoch(self, 
                  epoch: int,
                  train_loss: float,
                  val_loss: float,
                  loss_components: Optional[Dict[str, float]] = None,
                  similarity: Optional[float] = None,
                  lr: Optional[float] = None):
        """记录一个epoch的信息"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 准备WandB日志
        wandb_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr if lr is not None else 0.0
        }
        
        if loss_components:
            for key in self.loss_components:
                if key in loss_components:
                    self.loss_components[key].append(loss_components[key])
                    wandb_metrics[f"train_{key}"] = loss_components[key]
        
        if similarity is not None:
            self.similarities.append(similarity)
            wandb_metrics["similarity"] = similarity
            
        if lr is not None:
            self.learning_rates.append(lr)
        
        # 记录到WandB
        if self.use_wandb:
            wandb.log(wandb_metrics, step=epoch)
    
    def finish(self):
        """结束WandB运行"""
        if self.use_wandb:
            wandb.finish()
    
    def save_logs(self):
        """保存日志到文件"""
        log_file = os.path.join(self.log_dir, 'training_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Epoch,Train_Loss,Val_Loss,InfoNCE,CORAL,Barlow,Similarity,LR\n")
            for i in range(len(self.train_losses)):
                line = f"{i+1},{self.train_losses[i]:.6f},{self.val_losses[i]:.6f}"
                
                # 损失组件
                for key in ['infonce', 'coral', 'barlow']:
                    if i < len(self.loss_components[key]):
                        line += f",{self.loss_components[key][i]:.6f}"
                    else:
                        line += ",0.0"
                
                # 相似度和学习率
                if i < len(self.similarities):
                    line += f",{self.similarities[i]:.6f}"
                else:
                    line += ",0.0"
                    
                if i < len(self.learning_rates):
                    line += f",{self.learning_rates[i]:.2e}"
                else:
                    line += ",0.0"
                
                f.write(line + "\n")
    
    def get_best_epoch(self) -> int:
        """获取最佳epoch"""
        if not self.val_losses:
            return 0
        return np.argmin(self.val_losses) + 1


class Trainer:
    """训练器类 - 集成Optuna和WandB支持"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config,
                 use_wandb: bool = False,
                 trial: Optional['optuna.Trial'] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)
        self.trial = trial  # Optuna试验对象
        
        # 创建损失函数
        self.criterion = create_loss_function(config)
        self.criterion.to(self.device)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建调度器
        self.scheduler = self._create_scheduler()
        
        # 早停
        self.early_stopping = None
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.patience,
                min_delta=config.training.min_delta
            )
        
        # 日志记录 - 集成WandB
        self.logger = TrainingLogger(
            config.paths.log_dir, 
            use_wandb=use_wandb and not trial  # 在Optuna试验中不使用WandB避免冲突
        )
        
        # 记录配置到WandB
        if use_wandb and not trial:
            self.logger.log_config(config)
        
        # 最佳模型记录
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(config.paths.checkpoint_dir, 'best_model.pth')
        
        print("训练器初始化完成")
    
    def _create_optimizer(self):
        """创建优化器"""
        return Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.training.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler_type == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_components = []
        
        progress_bar = tqdm(self.train_loader, desc='训练')
        
        for batch_idx, (ir_batch, vis_batch, _) in enumerate(progress_bar):
            # 移动到设备
            ir_batch = ir_batch.to(self.device, non_blocking=True)
            vis_batch = vis_batch.to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            ir_proj = self.model(ir_batch)
            vis_proj = self.model(vis_batch)
            
            # 计算损失
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_dict = self.criterion(ir_proj, vis_proj)
                epoch_components.append(loss_dict)
            else:
                loss = self.criterion(ir_proj, vis_proj)
                loss_dict = {'total': loss.item()}
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # 更新进度条
            if isinstance(self.criterion, CombinedLoss):
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'InfoNCE': f'{loss_dict["infonce"]:.6f}',
                    'CORAL': f'{loss_dict["coral"]:.6f}',
                    'Barlow': f'{loss_dict["barlow"]:.6f}'
                })
            else:
                progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        
        # 计算平均损失组件
        avg_components = {}
        if epoch_components:
            for key in epoch_components[0].keys():
                avg_components[key] = np.mean([d[key] for d in epoch_components])
        
        return avg_loss, avg_components
    
    def validate_epoch(self) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        
        val_losses = []
        all_ir_proj = []
        all_vis_proj = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='验证')
            
            for ir_batch, vis_batch, _ in progress_bar:
                ir_batch = ir_batch.to(self.device, non_blocking=True)
                vis_batch = vis_batch.to(self.device, non_blocking=True)
                
                ir_proj = self.model(ir_batch)
                vis_proj = self.model(vis_batch)
                
                # 计算损失
                if isinstance(self.criterion, CombinedLoss):
                    loss, _ = self.criterion(ir_proj, vis_proj)
                else:
                    loss = self.criterion(ir_proj, vis_proj)
                
                val_losses.append(loss.item())
                
                # 收集特征用于相似度计算
                all_ir_proj.append(ir_proj.cpu())
                all_vis_proj.append(vis_proj.cpu())
                
                progress_bar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_val_loss = np.mean(val_losses)
        
        # 计算特征相似度
        all_ir_proj = torch.cat(all_ir_proj, dim=0)
        all_vis_proj = torch.cat(all_vis_proj, dim=0)
        similarity = compute_feature_similarity(all_ir_proj, all_vis_proj)
        
        return avg_val_loss, similarity
    
    def save_model(self, path: str, epoch: int, is_best: bool = False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            print(f"✅ 保存最佳模型: {path}")
        else:
            print(f"💾 保存检查点: {path}")
    
    def train(self):
        """主训练循环 - 集成Optuna剪枝和WandB记录"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"使用设备: {self.device}")
        print(f"WandB状态: {'✅启用' if self.logger.use_wandb else '❌禁用'}")
        print(f"Optuna剪枝: {'✅启用' if self.trial else '❌禁用'}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_loss, train_components = self.train_epoch()
            
            # 验证阶段
            val_loss, similarity = self.validate_epoch()
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            # 记录日志
            self.logger.log_epoch(
                epoch + 1, train_loss, val_loss, 
                train_components, similarity, current_lr
            )
            
            # Optuna剪枝检查
            if self.trial is not None:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    print(f"\n✂️ Optuna剪枝触发：第{epoch+1}轮停止试验")
                    self.logger.finish()
                    raise optuna.TrialPruned()
            
            # 打印结果
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs} (耗时: {epoch_time:.1f}s)")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  特征相似度: {similarity:.4f}")
            print(f"  学习率: {current_lr:.2e}")
            
            if train_components:
                print(f"  InfoNCE: {train_components['infonce']:.6f}")
                print(f"  CORAL: {train_components['coral']:.6f}")
                print(f"  Barlow: {train_components['barlow']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(self.best_model_path, epoch + 1, is_best=True)
                
                # 记录最佳模型到WandB
                if self.logger.use_wandb:
                    wandb.log({"best_val_loss": val_loss}, step=epoch+1)
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoint_dir, 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                self.save_model(checkpoint_path, epoch + 1)
            
            # 早停检查
            if self.early_stopping and self.early_stopping(val_loss):
                print(f"\n⏹️ 早停触发：验证损失连续{self.config.training.patience}轮未改善")
                break
        
        # 训练结束
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！")
        print(f"总耗时: {total_time/3600:.1f} 小时")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        # 结束WandB运行
        self.logger.finish()
        
        # 保存最终模型和日志
        final_model_path = os.path.join(self.config.paths.checkpoint_dir, 'final_model.pth')
        self.save_model(final_model_path, self.config.training.num_epochs)
        self.logger.save_logs()
        
        return self.best_val_loss


def create_hyperparameter_study(config, model_factory, data_loaders_factory, n_trials: int = 50):
    """创建Optuna超参数优化研究"""
    if not OPTUNA_AVAILABLE:
        raise ImportError("需要安装Optuna: pip install optuna")
    
    def suggest_hyperparameters(trial):
        """建议超参数"""
        return {
            # 学习率
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            
            # 批次大小
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            
            # 权重衰减
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            
            # 温度参数
            "temperature": trial.suggest_float("temperature", 0.01, 0.2),
            
            # 损失权重
            "lambda_coral": trial.suggest_float("lambda_coral", 0.001, 0.1, log=True),
            "lambda_barlow": trial.suggest_float("lambda_barlow", 0.001, 0.1, log=True),
            
            # 投影头配置
            "projection_hidden_dim": trial.suggest_categorical("projection_hidden_dim", [128, 256, 512]),
            "projection_output_dim": trial.suggest_categorical("projection_output_dim", [64, 128, 256]),
            
            # 调度器参数
            "scheduler_type": trial.suggest_categorical("scheduler_type", ["cosine", "step"]),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
        }
    
    def objective(trial):
        """优化目标函数"""
        # 获取建议的超参数
        hyperparams = suggest_hyperparameters(trial)
        
        # 创建配置副本并更新超参数
        trial_config = create_trial_config(config, hyperparams)
        
        # 创建模型和数据加载器
        model = model_factory(trial_config)
        train_loader, val_loader = data_loaders_factory(trial_config)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trial_config,
            use_wandb=False,  # 在Optuna中不使用WandB避免冲突
            trial=trial
        )
        
        # 训练模型
        try:
            best_val_loss = trainer.train()
            return best_val_loss
        except Exception as e:
            print(f"Trial {trial.number} 失败: {e}")
            raise optuna.TrialPruned()
    
    # 创建研究
    study = optuna.create_study(direction="minimize")
    
    # 执行优化
    study.optimize(objective, n_trials=n_trials)
    
    # 输出结果
    print("=" * 60)
    print("🎯 超参数优化完成")
    print("=" * 60)
    print(f"最佳试验: {study.best_trial.number}")
    print(f"最佳验证损失: {study.best_value:.6f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


def create_trial_config(base_config, hyperparams):
    """为试验创建配置"""
    from config import Config
    
    config = Config()
    
    # 复制基础配置
    config.training = base_config.training
    config.data = base_config.data
    config.model = base_config.model
    config.loss = base_config.loss
    config.paths = base_config.paths
    config.device = base_config.device
    
    # 应用超参数
    config.training.learning_rate = hyperparams["learning_rate"]
    config.training.batch_size = hyperparams["batch_size"]
    config.training.weight_decay = hyperparams["weight_decay"]
    config.training.scheduler_type = hyperparams["scheduler_type"]
    config.training.warmup_epochs = hyperparams["warmup_epochs"]
    
    config.loss.temperature = hyperparams["temperature"]
    config.loss.lambda_coral = hyperparams["lambda_coral"]
    config.loss.lambda_barlow = hyperparams["lambda_barlow"]
    
    config.model.projection_hidden_dim = hyperparams["projection_hidden_dim"]
    config.model.projection_output_dim = hyperparams["projection_output_dim"]
    
    return config


if __name__ == "__main__":
    # 测试训练器
    from config import get_config
    from data_loader import create_dataloaders
    from models import create_feature_extractor
    
    config = get_config()
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建模型
    model = create_feature_extractor(config, config.device)
    
    if model is not None:
        # 创建训练器
        trainer = Trainer(model, train_loader, val_loader, config)
        
        # 开始训练（测试一个epoch）
        config.training.num_epochs = 1
        trainer.train()
