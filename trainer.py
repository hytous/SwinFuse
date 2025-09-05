# -*- coding: utf-8 -*-
"""
训练器模块
管理训练过程、优化器、调度器等

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
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 记录列表
        self.train_losses = []
        self.val_losses = []
        self.loss_components = {'infonce': [], 'coral': [], 'barlow': []}
        self.similarities = []
        self.learning_rates = []
        
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
        
        if loss_components:
            for key in self.loss_components:
                if key in loss_components:
                    self.loss_components[key].append(loss_components[key])
        
        if similarity is not None:
            self.similarities.append(similarity)
            
        if lr is not None:
            self.learning_rates.append(lr)
    
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
    """训练器类"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)
        
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
        
        # 日志记录
        self.logger = TrainingLogger(config.paths.log_dir)
        
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
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
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
        print(f"最佳epoch: {self.logger.get_best_epoch()}")
        
        # 保存最终模型和日志
        final_model_path = os.path.join(self.config.paths.checkpoint_dir, 'final_model.pth')
        self.save_model(final_model_path, self.config.training.num_epochs)
        self.logger.save_logs()
        
        return self.best_val_loss


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
