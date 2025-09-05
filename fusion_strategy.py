# -*- coding: utf-8 -*-
"""
融合策略模块 (fusion_strategy.py) - 占位实现

功能说明:
    为SwinFuse网络提供融合策略函数的占位实现，避免导入错误

背景:
    原始SwinFuse网络中的fusion方法需要fusion_strategy模块
    但在本项目中只使用编码器部分进行特征提取，不需要实际融合功能
    因此提供占位实现以保持代码兼容性

主要内容:
    - attention_fusion_weight: 注意力融合权重函数 (占位)
        * 原本用于IR和VIS特征的注意力加权融合
        * 当前直接返回第一个特征，不影响特征提取
    - average_fusion: 平均融合 (占位)
    - maximum_fusion: 最大值融合 (占位)

实现特点:
    - 所有函数都是占位实现
    - 不会在特征提取过程中被实际调用
    - 仅为保持与原始SwinFuse代码的兼容性

使用说明:
    由于本项目专注于特征提取而非图像融合，这些函数不会被执行
    FeatureExtractor类只使用SwinFuse的编码器组件

注意事项:
    - 如需实际融合功能，需要实现具体的融合算法
    - 当前实现仅用于避免导入错误
    - 不影响特征提取器的正常工作

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import torch
import torch.nn as nn


def attention_fusion_weight(feature1, feature2, p_type):
    """
    注意力融合权重函数（占位实现）
    
    由于本项目只使用编码器进行特征提取，不需要实际融合功能，
    这里提供一个简单的占位实现以避免导入错误。
    
    Args:
        feature1: 第一个特征
        feature2: 第二个特征  
        p_type: 融合类型
        
    Returns:
        融合后的特征（这里直接返回第一个特征）
    """
    # 占位实现：直接返回第一个特征
    # 在实际的特征提取过程中不会调用到这个函数
    return feature1


# 为了保持与原始代码的兼容性，也可以定义其他可能需要的融合策略
def average_fusion(feature1, feature2):
    """平均融合（占位）"""
    return (feature1 + feature2) / 2.0


def maximum_fusion(feature1, feature2):
    """最大值融合（占位）"""
    return torch.maximum(feature1, feature2)


# 如果有其他融合策略的引用，可以在这里添加占位实现
