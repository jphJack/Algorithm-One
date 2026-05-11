# VIBE 双模态生物特征识别网络

## 项目概述

VIBE（Vein and Palm Print Biometric Enhancement）网络是一个基于掌纹和掌静脉双模态的生物特征识别系统。该网络采用双层递进式混合专家（Mixture of Experts, MoE）架构，通过多频域特征提取和跨模态融合机制，实现高精度的个人身份识别。

### 主要特性

- **双模态融合**：同时处理掌纹图像和掌静脉图像
- **多尺度特征提取**：骨干网络输出 Stage 3/4/5 三层多尺度特征，通过 Reducer 统一通道后融合
- **多频域特征增强**：通过高频、中频、低频三个专家网络分别提取不同频率的特征信息
- **跨模态交互融合**：利用交叉注意力、多尺度卷积和通道交互三种机制实现掌纹和掌静脉特征的深度融合
- **ArcFace 判别头**：采用 ArcFace (Additive Angular Margin) 损失增强类间可分性
- **负载均衡损失**：MoE 模块引入负载均衡损失，防止专家坍塌
- **轻量化设计**：采用深度可分离卷积和空洞卷积等技术降低计算复杂度

### 训练结果

| 数据集 | 类别数 | 训练准确率 | 测试准确率 | 备注 |
|--------|--------|-----------|-----------|------|
| HandsData | 290 | 100% | 99.59% | 128×128 |
| CUMT | - | 100% | 99.79% | - |
| CUMT2 | 532 | 100% | 99.62% | 160×160 |
| QH | 500 | 100% | 99.83% | 128×128 |
| TJ | 600 | 100% | 99.98% | 128×128 (第30轮) |

## 网络结构

### 整体架构

```
输入层
    │
    ├── 掌纹图像 (128×128×3)
    └── 掌静脉图像 (128×128×3)
    │
    ▼
双支流骨干网络 (DualStreamBackbone)
    ├── 掌纹骨干网络 (独立参数) → 多尺度特征提取器
    └── 掌静脉骨干网络 (独立参数) → 多尺度特征提取器
    │  输出: [B, 256, H/32, W/32]
    ▼
MoE特征增强模块 (MoEEnhancement) × 2
    ├── 高频专家 - 拉普拉斯边缘检测 + 深度可分离卷积
    ├── 中频专家 - 多尺度空洞卷积 + 方向感知卷积
    └── 低频专家 - 编码器-解码器 + 低通滤波
    │  门控网络: 自适应加权融合
    ▼
MoE融合模块 (MoEFusion)
    ├── 跨注意力融合专家 - 双向交叉注意力 + 残差连接
    ├── 多尺度卷积融合专家 - 1×1/3×3/5×5 并行卷积
    └── 通道交互融合专家 - 通道注意力校准
    │  门控网络: 双模态联合门控
    ▼
分类网络 (Classifier)
    ├── 全局平均池化
    ├── 嵌入层 (Linear + BatchNorm + Dropout)
    └── ArcFace 判别头 (ArcMarginProduct)
    │
    ▼
输出层 (N类分类)
```

### 模块详细说明

#### 1. 双支流骨干网络 (DualStreamBackbone)

采用轻量化 CNN 架构，分别处理掌纹和掌静脉图像。每个支流包含 5 个卷积阶段，输出 Stage 3、4、5 的多尺度特征图，通过 Reducer（1×1 卷积）统一通道数后，上采样到同一分辨率并拼接，再经投影层融合为统一特征。

**单支流骨干网络结构**：

| 阶段 | 输入尺寸 | 输出通道 | 输出尺寸 | 空间缩减 |
|------|---------|---------|---------|----------|
| Stage 1 | 128×128 | 32 | 64×64 | 2× |
| Stage 2 | 64×64 | 64 | 32×32 | 2× |
| Stage 3 | 32×32 | 128 | 16×16 | 2× |
| Stage 4 | 16×16 | 256 | 8×8 | 2× |
| Stage 5 | 8×8 | 256 | 4×4 | 2× |

每个 Stage 包含 2 个卷积块（Conv2d → BatchNorm2d → ReLU），第一个 stride=2 下采样，第二个 stride=1 精炼特征。

**多尺度特征提取器 (MultiScaleFeatureExtractor)**：
- 从 Stage 3/4/5 提取特征图
- Reducer 将各层通道统一为 64
- 上采样到 Stage 3 的分辨率后拼接
- 经 1×1 + 3×3 卷积投影到 feature_dim (256)

#### 2. MoE特征增强模块 (MoEEnhancement)

为每种模态构建独立的专家网络池，包含三个并行专家和一个门控网络：

- **高频专家 (HighFreqExpert)**：使用固定拉普拉斯核提取边缘特征，结合两个深度可分离卷积增强块（残差连接）提取细节纹理
- **中频专家 (MidFreqExpert)**：ASPP 多尺度空洞卷积（d=1,2,4）+ 方向感知卷积（1×7 和 7×1），残差连接
- **低频专家 (LowFreqExpert)**：编码器-解码器结构（MaxPool 下采样 + 瓶颈层 + 上采样 + skip connection）+ 低通滤波（5×5 平均池化），可学习参数 α 融合两支路

**门控网络 (GateNetwork)**：全局平均池化 → FC(256→64) → ReLU → FC(64→3) → Softmax

**负载均衡损失**：`L_lb = N * Σ(f_i²) - 1`，其中 f_i 为各专家平均权重，防止专家坍塌。

#### 3. MoE融合模块 (MoEFusion)

构建跨模态交互专家池，包含三个异构专家和一个双模态联合门控网络：

- **跨注意力融合专家 (CrossAttentionExpert)**：双向交叉注意力（掌纹→静脉、静脉→掌纹），拼接原始特征和交叉特征后 1×1 卷积融合，残差连接 `0.5×(f_p + f_v)`
- **多尺度卷积融合专家 (MultiScaleConvExpert)**：拼接双模态特征 → 1×1 降维 → 1×1/3×3/5×5 并行卷积 → 逐元素相加融合
- **通道交互融合专家 (ChannelInteractionExpert)**：全局池化 → MLP 生成通道权重 → 互相校准对方模态 → 相加融合

**融合门控网络 (FusionGateNetwork)**：双模态全局池化拼接 → FC(512→128) → ReLU → FC(128→3) → Softmax

#### 4. 分类网络 (Classifier)

采用 ArcFace 判别头增强类间可分性：

- 全局平均池化 → 嵌入层 Linear(256→256) → BatchNorm1d → Dropout(0.5)
- ArcMarginProduct：对嵌入向量和类别权重做 L2 归一化，计算余弦相似度，对目标类别施加角度惩罚 m=0.5，缩放因子 s=30.0

## 数据规格

| 数据集 | 类别数 | 图像尺寸 | 数据路径 |
|--------|--------|---------|---------|
| HandsData | 290 | 128×128 | HandsData/ |
| CASIA | 200 | 128×128 | data2/CASIA/ |
| QH | 500 | 128×128 | data2/QH/ |
| TJ | 600 | 128×128 | data2/TJ/ |
| CUMT2 | 532 | 192×192 | data2/CUMT2/ |

## 数据增强

训练时对掌纹和掌静脉图像应用**同步几何变换**和**独立颜色增强**：

- **同步几何变换**：随机裁剪（相同位置）、随机旋转（相同角度，±15°）
- **颜色抖动**：亮度 ±0.2、对比度 ±0.2、饱和度 ±0.2、色调 ±0.1
- **归一化**：ImageNet 均值 [0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225]

## 训练策略

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-4) |
| 学习率调度 | Warmup (5 epochs) + Cosine Annealing (min_lr=1e-6) |
| 损失函数 | CrossEntropy + Label Smoothing (0.05) + Load Balancing (weight=0.01) |
| 梯度裁剪 | max_norm=1.0 |
| Batch Size | 32 |
| Epochs | 100 |
| 随机种子 | 42 (确定性模式) |

## 项目结构

```
.
├── models/
│   ├── __init__.py
│   ├── backbone.py          # 双支流骨干网络 + 多尺度特征提取器
│   ├── classifier.py        # 分类网络 (ArcFace判别头)
│   ├── moe_enhancement.py   # MoE特征增强模块 (含负载均衡)
│   ├── moe_fusion.py        # MoE融合模块 (含负载均衡)
│   └── vibe_net.py          # VIBE主网络
├── utils/
│   └── __init__.py
├── config.py                # 配置文件 (多数据集支持)
├── dataset.py               # 数据集加载 + 数据增强
├── main.py                  # 主程序入口
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── test_model.py            # 模型测试
├── analyze_experts.py       # 专家分析脚本
├── network_architecture.md  # 网络架构详细文档
└── README.md                # 项目说明文档
```

## 使用方法

### 训练

```bash
python main.py --mode train --dataset <dataset_name>
```

可选数据集：`HandsData`（默认）、`CASIA`、`QH`、`TJ`、`CUMT2`

### 测试

```bash
# 使用默认 checkpoint
python test.py --dataset HandsData

# 指定 checkpoint 路径
python test.py --dataset QH --checkpoint checkpoints4/QH/best_model.pth
```

### 演示

```bash
python main.py --mode demo
```

## 依赖环境

- Python 3.7+
- PyTorch 1.8+
- NumPy
- OpenCV (cv2)
- Matplotlib
- scikit-learn
- tqdm

## 论文引用

如需引用本项目，请参考相关学术论文。