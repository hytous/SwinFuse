# 🚀 快速开始指南

## 项目设置

1. **验证环境**:
```bash
python run.py --mode validate
```

2. **开始训练**:
```bash
python run.py --mode train
```

3. **自定义训练参数**:
```bash
python run.py --mode train --data-dir /path/to/RoadScene --epochs 50 --batch-size 16
```

## 主要文件说明

- `run.py` - 项目启动脚本，推荐入口
- `main.py` - 直接训练入口
- `config.py` - 所有配置项
- `models.py` - 特征提取器模型
- `trainer.py` - 训练逻辑
- `losses.py` - 损失函数实现

## 配置修改

在 `config.py` 中修改关键参数：

```python
# 数据路径
data_root = "/path/to/your/RoadScene"

# 预训练模型路径  
pretrained_path = "SwinFuse_model/Final_epoch_50_Mon_Feb_14_17_37_05_2022_1e3.model"

# 训练参数
num_epochs = 100
batch_size = 32
learning_rate = 1e-4
```

## 输出文件

训练完成后会生成：
- `outputs/checkpoints/` - 模型检查点
- `outputs/logs/` - 训练日志
- `outputs/plots/` - 可视化图表

## 常见问题

1. **内存不足**: 减小 batch_size
2. **GPU不可用**: 自动使用CPU训练  
3. **数据路径错误**: 检查config.py中的data_root设置
