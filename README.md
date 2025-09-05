# SwinFuse特征提取器微调项目

基于SwinFuse的特征提取器微调项目，用于IR-可见光图像配准任务。本项目采用模块化设计，支持多种先进的损失函数和完整的训练管道。

## 📋 项目概述

本项目基于SwinFuse（Swin Transformer融合网络）的编码器部分，通过在RoadScene数据集上进行微调，实现红外-可见光图像的特征对齐，用于图像配准任务。

### 主要特性

- 🔥 **模块化设计**: 清晰的代码结构，易于维护和扩展
- 🎯 **先进损失函数**: 支持InfoNCE、Deep CORAL、Barlow Twins等
- 📊 **完整训练管道**: 包含数据加载、训练、验证、可视化
- ⚡ **GPU加速**: 支持CUDA训练，自动混合精度
- 📈 **实时监控**: 训练过程可视化，早停机制
- 🔧 **灵活配置**: 统一的配置管理系统

## 🏗️ 项目结构

```
SwinFuse/
├── config.py              # 统一配置管理
├── data_loader.py          # 数据集加载和预处理
├── models.py               # 特征提取器模型定义
├── losses.py               # 损失函数实现
├── trainer.py              # 训练逻辑
├── main.py                 # 主入口程序
├── visualization.py        # 可视化工具
├── utils_clean.py          # 通用工具函数
├── run.py                  # 项目启动脚本
├── net.py                  # 原始SwinFuse网络
├── README_NEW.md           # 项目文档（本文件）
└── SwinFuse_model/         # 预训练模型目录
    └── Final_epoch_50_Mon_Feb_14_17_37_05_2022_1e3.model
```

## 🚀 快速开始

### 环境要求

- Python >= 3.7
- PyTorch >= 1.6.0
- CUDA >= 10.1 (可选，用于GPU加速)

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib seaborn tqdm numpy
```

### 数据准备

1. 准备RoadScene数据集，确保目录结构如下：
```
RoadScene/
├── train/
│   ├── ir/          # 红外图像
│   └── visible/     # 可见光图像
└── val/
    ├── ir/
    └── visible/
```

2. 确保红外和可见光图像具有相同的文件名用于配对

### 运行项目

#### 1. 验证环境和配置

```bash
python run.py --mode validate
```

#### 2. 开始训练

```bash
# 使用默认配置训练
python run.py --mode train

# 自定义参数训练
python run.py --mode train --data-dir /path/to/RoadScene --epochs 100 --batch-size 32 --lr 1e-4

# 从检查点恢复训练
python run.py --mode train --resume
```

#### 3. 测试模型

```bash
# 测试最佳模型
python run.py --mode test --checkpoint best

# 测试指定检查点
python run.py --mode test --checkpoint /path/to/checkpoint.pth
```

#### 4. 清理临时文件

```bash
python run.py --mode clean
```

## ⚙️ 配置说明

项目使用 `config.py` 进行统一配置管理，主要配置项包括：

### 数据配置 (DataConfig)
- `data_root`: 数据集根目录
- `batch_size`: 批次大小
- `num_workers`: 数据加载进程数
- `image_size`: 图像尺寸
- `train_split`: 训练集比例

### 训练配置 (TrainingConfig)
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减
- `gradient_clip_norm`: 梯度裁剪阈值
- `mixed_precision`: 是否使用混合精度

### 模型配置 (ModelConfig)
- `pretrained_path`: 预训练模型路径
- `freeze_encoder`: 是否冻结编码器
- `projection_dim`: 投影头维度
- `dropout_rate`: Dropout比率

### 损失配置 (LossConfig)
- `loss_type`: 损失函数类型 ('combined', 'infonce', 'coral', 'barlow')
- `temperature`: InfoNCE温度参数
- `coral_weight`: Deep CORAL权重
- `barlow_weight`: Barlow Twins权重

## 📊 损失函数

项目实现了多种先进的损失函数：

### 1. InfoNCE Loss
基于对比学习的损失函数，用于学习判别性特征表示：

```python
loss = -log(exp(sim_pos / τ) / Σ exp(sim_i / τ))
```

### 2. Deep CORAL Loss
用于对齐不同模态间的特征分布：

```python
loss = ||Cov(f_ir) - Cov(f_vis)||²_F / (4 * d²)
```

### 3. Barlow Twins Loss
通过最大化互相关矩阵的对角元素，最小化非对角元素来学习特征：

```python
loss = Σᵢ(1 - C_ii)² + λ Σᵢ Σⱼ≠ᵢ C_ij²
```

### 4. Combined Loss
组合多种损失函数的加权和：

```python
loss = w₁ * InfoNCE + w₂ * DeepCORAL + w₃ * BarlowTwins
```

## 📈 训练监控

项目提供了完整的训练监控功能：

### 实时指标
- 训练/验证损失
- 学习率变化
- GPU内存使用情况
- 训练速度 (it/s)

### 可视化
- 损失曲线图
- 特征相似度热图
- 学习率调度图

### 早停机制
- 基于验证损失的早停
- 模型检查点自动保存
- 最佳模型选择

## 🔧 高级功能

### 1. 混合精度训练
使用PyTorch的自动混合精度加速训练：

```python
# 自动启用，无需额外配置
config.training.mixed_precision = True
```

### 2. 学习率调度
支持多种学习率调度策略：

```python
# 余弦退火调度
config.training.scheduler_type = 'cosine'
config.training.scheduler_params = {'T_max': 100}
```

### 3. 数据增强
内置多种数据增强技术：

```python
# 随机翻转、旋转、颜色抖动等
config.data.augmentation = True
```

### 4. 梯度裁剪
防止梯度爆炸：

```python
config.training.gradient_clip_norm = 1.0
```

## 📝 使用示例

### 基本训练示例

```python
from config import Config
from trainer import Trainer
from models import FeatureExtractor
from data_loader import create_dataloaders

# 加载配置
config = Config()

# 创建数据加载器
train_loader, val_loader = create_dataloaders(config)

# 创建模型
model = FeatureExtractor(config)

# 创建训练器
trainer = Trainer(model, config)

# 开始训练
trainer.train(train_loader, val_loader)
```

### 自定义损失函数

```python
from losses import BaseLoss

class CustomLoss(BaseLoss):
    def forward(self, features_ir, features_vis):
        # 实现自定义损失逻辑
        pass

# 在配置中指定
config.loss.custom_loss_class = CustomLoss
```

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 启用混合精度训练
   - 减少 `num_workers`

2. **训练不收敛**
   - 降低学习率
   - 检查数据预处理
   - 调整损失函数权重

3. **GPU利用率低**
   - 增加 `batch_size`
   - 增加 `num_workers`
   - 检查数据加载瓶颈

### 调试模式

```bash
# 启用详细日志
python run.py --mode train --log-level DEBUG

# 验证单个批次
python -c "
from data_loader import create_dataloaders
from config import Config
config = Config()
train_loader, _ = create_dataloaders(config)
batch = next(iter(train_loader))
print('Batch shape:', batch[0].shape, batch[1].shape)
"
```

## 📚 API参考

### 核心类

#### Config
统一配置管理类，包含所有配置项。

#### FeatureExtractor
特征提取器模型，基于SwinFuse编码器。

#### Trainer
训练管理器，负责训练循环、验证、保存等。

#### DataLoader
数据加载器，处理RoadScene数据集。

### 主要函数

#### create_dataloaders()
创建训练和验证数据加载器。

#### setup_logging()
设置日志系统。

#### print_system_info()
打印系统和GPU信息。

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目基于原SwinFuse项目进行重构和扩展。

## 🙏 致谢

- SwinFuse原始项目作者
- PyTorch团队
- 开源社区贡献者

## 📞 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**注意**: 这是一个重构后的现代化版本，如需使用原始代码，请参考项目中的原始文件。
