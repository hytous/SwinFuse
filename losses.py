# -*- coding: utf-8 -*-
"""
损失函数模块
实现InfoNCE、Deep CORAL、Barlow Twins等先进损失函数

作者: 基于SwinFuse项目重构  
日期: 2025年9月
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from abc import ABC, abstractmethod


class BaseLoss(ABC, nn.Module):
    """损失函数基类"""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    @abstractmethod
    def forward(self, z_ir: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        pass


class InfoNCELoss(BaseLoss):
    """InfoNCE对比损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__("InfoNCE")
        self.temperature = temperature
        
    def forward(self, z_ir: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ir: 红外特征 [B, D]
            z_vis: 可见光特征 [B, D]
        Returns:
            loss: InfoNCE损失
        """
        batch_size = z_ir.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_ir, z_vis.T) / self.temperature
        
        # 标签为对角线位置
        labels = torch.arange(batch_size, device=z_ir.device)
        
        # 双向对比损失
        loss_ir2vis = F.cross_entropy(sim_matrix, labels)
        loss_vis2ir = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_ir2vis + loss_vis2ir) / 2.0


class DeepCORALLoss(BaseLoss):
    """Deep CORAL分布对齐损失"""
    
    def __init__(self):
        super().__init__("CORAL")
        
    def forward(self, z_ir: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ir: 红外特征 [B, D]
            z_vis: 可见光特征 [B, D]
        Returns:
            loss: CORAL损失
        """
        batch_size = z_ir.size(0)
        d = z_ir.size(1)
        
        # 计算协方差矩阵
        cov_ir = self._compute_covariance(z_ir)
        cov_vis = self._compute_covariance(z_vis)
        
        # Frobenius范数
        loss = torch.sum((cov_ir - cov_vis) ** 2) / (4.0 * d * d)
        
        return loss
    
    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """计算协方差矩阵"""
        # 中心化
        x_centered = x - x.mean(0, keepdim=True)
        
        # 协方差矩阵
        batch_size = x.size(0)
        cov = torch.matmul(x_centered.T, x_centered) / (batch_size - 1)
        
        return cov


class BarlowTwinsLoss(BaseLoss):
    """Barlow Twins去冗余损失"""
    
    def __init__(self, lambda_off_diag: float = 0.005):
        super().__init__("BarlowTwins")
        self.lambda_off_diag = lambda_off_diag
        
    def forward(self, z_ir: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ir: 红外特征 [B, D]
            z_vis: 可见光特征 [B, D]
        Returns:
            loss: Barlow Twins损失
        """
        batch_size = z_ir.size(0)
        
        # 计算交相关矩阵
        cross_corr = torch.matmul(z_ir.T, z_vis) / batch_size
        
        # 对角线损失（希望接近1）
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        
        # 非对角线损失（希望接近0）
        off_diag = self._off_diagonal(cross_corr).pow_(2).sum()
        
        loss = on_diag + self.lambda_off_diag * off_diag
        
        return loss
    
    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """获取非对角线元素"""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, 
                 temperature: float = 0.07,
                 lambda_coral: float = 0.02,
                 lambda_barlow: float = 0.01,
                 lambda_off_diag: float = 0.005):
        super().__init__()
        
        # 创建各个损失函数
        self.infonce_loss = InfoNCELoss(temperature)
        self.coral_loss = DeepCORALLoss()
        self.barlow_loss = BarlowTwinsLoss(lambda_off_diag)
        
        # 损失权重
        self.lambda_coral = lambda_coral
        self.lambda_barlow = lambda_barlow
        
        print(f"组合损失初始化:")
        print(f"  InfoNCE温度: {temperature}")
        print(f"  CORAL权重: {lambda_coral}")
        print(f"  Barlow权重: {lambda_barlow}")
        
    def forward(self, z_ir: torch.Tensor, z_vis: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_ir: 红外特征 [B, D]
            z_vis: 可见光特征 [B, D]
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        # 计算各项损失
        loss_infonce = self.infonce_loss(z_ir, z_vis)
        loss_coral = self.coral_loss(z_ir, z_vis)
        loss_barlow = self.barlow_loss(z_ir, z_vis)
        
        # 组合损失
        total_loss = (loss_infonce + 
                     self.lambda_coral * loss_coral + 
                     self.lambda_barlow * loss_barlow)
        
        # 损失详情
        loss_dict = {
            'total': total_loss.item(),
            'infonce': loss_infonce.item(),
            'coral': loss_coral.item(),
            'barlow': loss_barlow.item()
        }
        
        return total_loss, loss_dict


def create_loss_function(config) -> nn.Module:
    """根据配置创建损失函数"""
    loss_type = config.loss.loss_type.lower()
    
    if loss_type == "combined":
        loss_fn = CombinedLoss(
            temperature=config.loss.temperature,
            lambda_coral=config.loss.lambda_coral,
            lambda_barlow=config.loss.lambda_barlow,
            lambda_off_diag=config.loss.lambda_off_diag
        )
    elif loss_type == "infonce":
        loss_fn = InfoNCELoss(temperature=config.loss.temperature)
    elif loss_type == "coral":
        loss_fn = DeepCORALLoss()
    elif loss_type == "barlow":
        loss_fn = BarlowTwinsLoss(lambda_off_diag=config.loss.lambda_off_diag)
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")
    
    print(f"创建损失函数: {loss_type}")
    return loss_fn


def compute_feature_similarity(z_ir: torch.Tensor, z_vis: torch.Tensor) -> float:
    """计算特征相似度"""
    with torch.no_grad():
        # 余弦相似度
        cosine_sim = F.cosine_similarity(z_ir, z_vis, dim=1)
        return cosine_sim.mean().item()


def test_loss_functions():
    """测试损失函数"""
    print("测试损失函数...")
    
    # 创建测试数据
    batch_size = 8
    feature_dim = 128
    
    torch.manual_seed(42)
    z_ir = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    z_vis = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    
    # 测试各个损失
    losses = [
        InfoNCELoss(),
        DeepCORALLoss(),
        BarlowTwinsLoss(),
        CombinedLoss()
    ]
    
    for loss_fn in losses:
        if isinstance(loss_fn, CombinedLoss):
            total_loss, loss_dict = loss_fn(z_ir, z_vis)
            print(f"{loss_fn.__class__.__name__}: {loss_dict}")
        else:
            loss_value = loss_fn(z_ir, z_vis)
            print(f"{loss_fn.__class__.__name__}: {loss_value.item():.6f}")
    
    # 测试相似度
    similarity = compute_feature_similarity(z_ir, z_vis)
    print(f"特征相似度: {similarity:.6f}")


if __name__ == "__main__":
    test_loss_functions()
