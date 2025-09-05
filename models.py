# -*- coding: utf-8 -*-
"""
模型定义模块 (models.py)

功能说明:
    基于SwinFuse的特征提取器模型定义，用于红外-可见光图像特征对齐

主要内容:
    - ProjectionHead: 投影头模块
        * 全局平均池化 + MLP
        * 用于对比学习的特征投影
        * L2归一化输出
    - FeatureExtractor: 特征提取器主模型
        * 复制SwinFuse编码器组件 (patch_embed, EN1_0, EN2_0, EN3_0)
        * 添加投影头进行特征降维
        * 支持冻结/解冻编码器参数
    - load_pretrained_swinfuse: 加载预训练SwinFuse模型
    - create_feature_extractor: 工厂函数创建特征提取器

模型架构:
    输入 -> SwinFuse编码器 -> 投影头 -> L2归一化特征
    [B,1,224,224] -> [B,96,H,W] -> [B,128] (归一化)

使用方法:
    from models import create_feature_extractor
    model = create_feature_extractor(config, device)

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from net import SwinFuse
from typing import Tuple, Optional
import numpy as np


class ProjectionHead(nn.Module):
    """投影头模块，用于对比学习"""
    
    def __init__(self, 
                 input_dim: int = 96, 
                 hidden_dim: int = 256, 
                 output_dim: int = 128,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            # 全局平均池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 第一层
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第二层
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            z: L2归一化的投影特征 [B, output_dim]
        """
        z = self.projection(x)
        z = F.normalize(z, p=2, dim=1)  # L2归一化
        return z


class FeatureExtractor(nn.Module):
    """特征提取器，基于SwinFuse编码器"""
    
    def __init__(self, swinfuse_model: SwinFuse, config):
        """
        Args:
            swinfuse_model: 预训练的SwinFuse模型
            config: 配置对象
        """
        super().__init__()
        
        # 复制编码器组件
        self._copy_encoder_components(swinfuse_model)
        
        # 添加投影头
        self.projection_head = ProjectionHead(
            input_dim=config.model.projection_input_dim,
            hidden_dim=config.model.projection_hidden_dim,
            output_dim=config.model.projection_output_dim
        )
        
        # 冻结部分层（可选）
        self._freeze_layers = False
        
        print(f"特征提取器初始化完成:")
        print(f"  编码器输出维度: {config.model.projection_input_dim}")
        print(f"  投影头输出维度: {config.model.projection_output_dim}")
    
    def _copy_encoder_components(self, swinfuse_model: SwinFuse):
        """复制编码器组件"""
        # Patch embedding
        self.patch_embed = swinfuse_model.patch_embed
        
        # Position embedding
        self.ape = swinfuse_model.ape
        if self.ape and hasattr(swinfuse_model, 'absolute_pos_embed'):
            self.absolute_pos_embed = swinfuse_model.absolute_pos_embed
        
        self.pos_drop = swinfuse_model.pos_drop
        
        # 编码器阶段
        self.EN1_0 = swinfuse_model.EN1_0
        self.EN2_0 = swinfuse_model.EN2_0
        self.EN3_0 = swinfuse_model.EN3_0
        
        # 归一化层
        self.norm = swinfuse_model.norm
    
    def freeze_encoder(self, freeze: bool = True):
        """冻结/解冻编码器参数"""
        self._freeze_layers = freeze
        
        components = [
            self.patch_embed, self.pos_drop,
            self.EN1_0, self.EN2_0, self.EN3_0, self.norm
        ]
        
        if hasattr(self, 'absolute_pos_embed'):
            self.absolute_pos_embed.requires_grad = not freeze
        
        for component in components:
            for param in component.parameters():
                param.requires_grad = not freeze
        
        print(f"编码器参数 {'冻结' if freeze else '解冻'}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入图像 [B, 1, H, W]
            return_features: 是否返回中间特征
        Returns:
            projections: 投影特征 [B, projection_dim]
            features: 原始特征 [B, C, H', W'] (可选)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Position embedding
        if self.ape and hasattr(self, 'absolute_pos_embed'):
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        # 编码器阶段（带残差连接）
        x1 = self.EN1_0(x) + x
        x2 = self.EN2_0(x1) + x1
        x3 = self.EN3_0(x2) + x2
        
        # 归一化
        x3 = self.norm(x3)
        
        # 转换为图像格式 [B, L, C] -> [B, C, H, W]
        features = self._reshape_features(x3)
        
        # 投影头
        projections = self.projection_head(features)
        
        if return_features:
            return projections, features
        else:
            return projections
    
    def _reshape_features(self, x: torch.Tensor) -> torch.Tensor:
        """重塑特征张量"""
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return x
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """获取特征维度信息"""
        return (
            self.projection_head.projection[-1].out_features,  # 投影维度
            self.EN3_0.norm1.normalized_shape[0]  # 编码器输出维度
        )


def load_pretrained_swinfuse(model_path: str, device: str = 'cpu') -> Optional[SwinFuse]:
    """加载预训练的SwinFuse模型"""
    print(f"正在加载预训练模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    try:
        # 创建模型
        model = SwinFuse(in_chans=1, out_chans=1)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ 预训练模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None


def create_feature_extractor(config, device: str = 'cpu', move_to_device: bool = True) -> Optional[FeatureExtractor]:
    """创建特征提取器
    
    Args:
        config: 配置对象
        device: 设备
        move_to_device: 是否立即移动到设备（多GPU情况下由trainer处理）
    """
    # 加载预训练模型
    swinfuse_model = load_pretrained_swinfuse(
        config.paths.pretrained_model_path, 
        device
    )
    
    if swinfuse_model is None:
        return None
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(swinfuse_model, config)
    
    # 仅在需要时移动到设备
    if move_to_device:
        feature_extractor.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in feature_extractor.parameters())
    trainable_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    
    print(f"特征提取器参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    return feature_extractor


if __name__ == "__main__":
    # 测试模型创建
    from config import get_config
    
    config = get_config()
    device = torch.device(config.device)
    
    # 创建特征提取器
    feature_extractor = create_feature_extractor(config, device)
    
    if feature_extractor is not None:
        # 测试前向传播
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 224, 224).to(device)
        
        with torch.no_grad():
            projections = feature_extractor(test_input)
            print(f"测试输出形状: {projections.shape}")
            
            # 测试返回特征
            projections, features = feature_extractor(test_input, return_features=True)
            print(f"投影形状: {projections.shape}")
            print(f"特征形状: {features.shape}")
