# -*- coding: utf-8 -*-
"""
可视化模块
提供训练过程和结果的可视化功能

作者: 基于SwinFuse项目重构
日期: 2025年9月
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn风格
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: List[float],
                           loss_components: Optional[Dict[str, List[float]]] = None,
                           similarities: Optional[List[float]] = None,
                           learning_rates: Optional[List[float]] = None):
        """绘制训练曲线"""
        
        # 确定子图数量
        n_plots = 2  # 基础：总损失 + 相似度
        if loss_components:
            n_plots += 1
        if learning_rates:
            n_plots += 1
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 总损失曲线
        axes[0].plot(epochs, train_losses, label='训练损失', color='#1f77b4', linewidth=2)
        axes[0].plot(epochs, val_losses, label='验证损失', color='#ff7f0e', linewidth=2)
        axes[0].set_title('训练和验证损失', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 损失组件
        if loss_components:
            colors = ['#2ca02c', '#d62728', '#9467bd']
            for i, (name, values) in enumerate(loss_components.items()):
                if name != 'total' and values:
                    axes[1].plot(epochs[:len(values)], values, 
                               label=name.upper(), color=colors[i % len(colors)], linewidth=2)
            
            axes[1].set_title('损失组件分解', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('训练轮次')
            axes[1].set_ylabel('损失值')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        # 3. 特征相似度
        if similarities:
            axes[2].plot(epochs[:len(similarities)], similarities, 
                        label='余弦相似度', color='#17becf', linewidth=2, marker='o', markersize=4)
            axes[2].set_title('特征相似度趋势', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('训练轮次')
            axes[2].set_ylabel('余弦相似度')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        
        # 4. 学习率曲线
        if learning_rates:
            axes[3].plot(epochs[:len(learning_rates)], learning_rates, 
                        label='学习率', color='#bcbd22', linewidth=2)
            axes[3].set_title('学习率变化', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('训练轮次')
            axes[3].set_ylabel('学习率')
            axes[3].set_yscale('log')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        else:
            # 显示训练统计
            best_train_loss = min(train_losses)
            best_val_loss = min(val_losses)
            best_epoch = np.argmin(val_losses) + 1
            
            stats_text = f'''训练统计信息
            
最佳训练损失: {best_train_loss:.6f}
最佳验证损失: {best_val_loss:.6f}
最佳轮次: {best_epoch}
总训练轮次: {len(train_losses)}

收敛趋势: {'良好' if val_losses[-1] < val_losses[0] else '需要调整'}'''
            
            axes[3].text(0.1, 0.5, stats_text, fontsize=12, 
                        verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            axes[3].set_xlim(0, 1)
            axes[3].set_ylim(0, 1)
            axes[3].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"训练曲线已保存: {save_path}")
        
        plt.show()
    
    def plot_loss_comparison(self, loss_data: Dict[str, List[float]]):
        """绘制不同损失函数的比较"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(loss_data)))
        
        for i, (name, values) in enumerate(loss_data.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name, color=colors[i], linewidth=2, marker='o', markersize=3)
        
        ax.set_title('损失函数比较', fontsize=16, fontweight='bold')
        ax.set_xlabel('训练轮次')
        ax.set_ylabel('损失值')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'loss_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"损失比较图已保存: {save_path}")
        
        plt.show()


def visualize_feature_similarity(ir_features: torch.Tensor, 
                                vis_features: torch.Tensor,
                                save_path: Optional[str] = None) -> float:
    """可视化特征相似度分布"""
    # 计算相似度矩阵
    with torch.no_grad():
        # 归一化特征
        ir_norm = F.normalize(ir_features, p=2, dim=1)
        vis_norm = F.normalize(vis_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(ir_norm, vis_norm.T)
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        # 对角线相似度（正样本对）
        diagonal_similarities = np.diag(similarity_matrix)
        
        # 非对角线相似度（负样本对）
        off_diagonal = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 相似度矩阵热图
    im = axes[0].imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('红外-可见光特征相似度矩阵', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('可见光特征索引')
    axes[0].set_ylabel('红外特征索引')
    plt.colorbar(im, ax=axes[0])
    
    # 2. 相似度分布直方图
    axes[1].hist(diagonal_similarities, bins=20, alpha=0.7, label='正样本对', color='green', density=True)
    axes[1].hist(off_diagonal, bins=20, alpha=0.7, label='负样本对', color='red', density=True)
    axes[1].set_title('特征相似度分布', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('余弦相似度')
    axes[1].set_ylabel('密度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 统计信息
    pos_mean = diagonal_similarities.mean()
    pos_std = diagonal_similarities.std()
    neg_mean = off_diagonal.mean()
    neg_std = off_diagonal.std()
    
    stats_text = f'''相似度统计
    
正样本对：
  均值: {pos_mean:.4f}
  标准差: {pos_std:.4f}
  
负样本对：
  均值: {neg_mean:.4f}
  标准差: {neg_std:.4f}
  
分离度: {pos_mean - neg_mean:.4f}'''
    
    axes[2].text(0.1, 0.5, stats_text, fontsize=11, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"特征相似度可视化已保存: {save_path}")
    
    plt.show()
    
    return pos_mean


def load_and_visualize_logs(log_file: str):
    """从日志文件加载并可视化训练过程"""
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return
    
    # 读取日志
    df = pd.read_csv(log_file)
    
    # 提取数据
    train_losses = df['Train_Loss'].tolist()
    val_losses = df['Val_Loss'].tolist()
    
    loss_components = {}
    if 'InfoNCE' in df.columns:
        loss_components['infonce'] = df['InfoNCE'].tolist()
    if 'CORAL' in df.columns:
        loss_components['coral'] = df['CORAL'].tolist()
    if 'Barlow' in df.columns:
        loss_components['barlow'] = df['Barlow'].tolist()
    
    similarities = df['Similarity'].tolist() if 'Similarity' in df.columns else None
    learning_rates = df['LR'].tolist() if 'LR' in df.columns else None
    
    # 创建可视化器
    save_dir = os.path.dirname(log_file)
    visualizer = TrainingVisualizer(save_dir)
    
    # 绘制训练曲线
    visualizer.plot_training_curves(
        train_losses, val_losses, loss_components, similarities, learning_rates
    )


if __name__ == "__main__":
    # 测试可视化功能
    # 生成模拟数据
    epochs = 30
    train_losses = [1.0 - 0.8 * np.exp(-i/10) + 0.1 * np.random.random() for i in range(epochs)]
    val_losses = [0.9 - 0.7 * np.exp(-i/12) + 0.15 * np.random.random() for i in range(epochs)]
    
    loss_components = {
        'infonce': [0.6 - 0.4 * np.exp(-i/8) + 0.05 * np.random.random() for i in range(epochs)],
        'coral': [0.3 - 0.2 * np.exp(-i/15) + 0.03 * np.random.random() for i in range(epochs)],
        'barlow': [0.1 - 0.05 * np.exp(-i/20) + 0.02 * np.random.random() for i in range(epochs)]
    }
    
    similarities = [0.3 + 0.6 * (1 - np.exp(-i/10)) + 0.05 * np.random.random() for i in range(epochs)]
    learning_rates = [1e-5 * (0.1 ** (i/20)) for i in range(epochs)]
    
    # 创建可视化器并绘制
    visualizer = TrainingVisualizer('test_vis')
    visualizer.plot_training_curves(train_losses, val_losses, loss_components, similarities, learning_rates)
    
    # 测试特征相似度可视化
    batch_size = 16
    feature_dim = 128
    ir_features = torch.randn(batch_size, feature_dim)
    vis_features = ir_features + 0.3 * torch.randn(batch_size, feature_dim)  # 添加噪声但保持相关性
    
    visualize_feature_similarity(ir_features, vis_features, 'test_vis/feature_similarity.png')
