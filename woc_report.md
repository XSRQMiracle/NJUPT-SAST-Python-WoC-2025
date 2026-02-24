# WOC报告
## 1.模型架构
### 1.1ResNet分类网络
修改版 ResNet，支持外部特征注入：
```
输入: [B, 3, 32, 32]
  ↓
Conv2d(3→16, 3×3) + BN + ReLU
  ↓
ResidualBlock(16→16, stride=1)  → f1 [B, 16, 32, 32]  ← + inj1
  ↓
ResidualBlock(16→32, stride=2)  → f2 [B, 32, 16, 16]  ← + inj2
  ↓
ResidualBlock(32→64, stride=2)  → f3 [B, 64,  8,  8]  ← + inj3
  ↓
AdaptiveAvgPool2d(1, 1) → Flatten → Linear(64→10)
  ↓
输出: logits [B, 10]
```
- **注入点**：每个block输出后加上CrossTaskBridge的注入信号
### 1.2 UNet去噪网络
带通道注意力的4级编码器-解码器：
```
输入: [B, 3, H, W]
  ↓
ShallowFeat: Conv2d(3→32)
  ↓
Encoder:
  enc1: 2×CAB(32)  → e1 [B, 32, H, W]                  提供给 Bridge1
  down1 + enc2: 2×CAB(64)  → e2 [B, 64, H/2, W/2]      提供给 Bridge2
  down2 + enc3: 2×CAB(128) → e3 [B, 128, H/4, W/4]     提供给 Bridge3
  down3 + enc4: 2×CAB(256) → e4
  down4
  ↓
Bottleneck: 4×CAB(256)
  ↓
Decoder: (Skip Connection + 1×1 Conv 通道减半 + CAB)
  dec4 → dec3 → dec2 → dec1
  ↓
Tail: Conv2d(32→3) + 全局残差 (output + identity)
  ↓
输出: denoised [B, 3, H, W]
```
- **CAB（Channel Attention Block）**：3×3 Conv + PReLU + 3×3 Conv + SE 通道注意力 + 残差
- **全局残差学习**：output = UNet(x) + x，学习残差
### 1.3 CrossTaskBridge（跨任务注入桥）
**3座桥**，连接UNet编码器和ResNet对应层：
```
Bridge1: UNet e1 [B,32,H,W]     → Conv1×1(32→16)  + BN → ×scale → inj1 [B,16,H,W]
Bridge2: UNet e2 [B,64,H/2,W/2] → Conv1×1(64→32)  + BN → ×scale → inj2 [B,32,H/2,W/2]
Bridge3: UNet e3 [B,128,H/4,W/4]→ Conv1×1(128→64) + BN → ×scale → inj3 [B,64,H/4,W/4]
```
---
## 2. 训练策略
### 2.1 两阶段训练（每个 Epoch）
**Phase 1 — 分类+CIFAR去噪（782 batch）**：
1. 对CIFAR-10图像添加随机高斯噪声（σ ~ U[0.05, 0.25]）
2. UNet完整前向：去噪+提取编码器特征
3. `scale_gradient(features, 0.1)` 缩放分类梯度回传到UNet
4. Bridge将UNet特征转换为注入信号
5. ResNet带注入做分类
6. `loss = cls_loss + warmup × λ × denoise_cifar_loss`
7. 同时更新ResNet+Bridges+UNet
**Phase 2 — SIDD去噪（72 batch）**：
1. UNet独立处理SIDD真实噪声图像
2. `loss = denoise_sidd_loss`
3. 仅更新UNet
### 2.2 梯度流向
| 损失来源 | 更新目标 | 梯度强度 |
|----------|----------|----------|
| `cls_loss` | ResNet + Bridges | 1.0× |
| `cls_loss` → Bridge → UNet | UNet encoder | **0.1×**（缩放） |
| `denoise_cifar_loss` | UNet | 1.0× |
| `denoise_sidd_loss` | UNet | 1.0× |

梯度缩放（`grad_scale_cls2unet = 0.1`）的作用：Phase 1 有 782 batch 的分类梯度流向 UNet，Phase 2 只有 72 batch 的去噪梯度。不缩放的话 UNet 会被分类任务主导，遗忘去噪能力。缩放后 UNet 主要忠于去噪，同时微弱感知分类需求。
### 2.3 超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 50 | 总训练轮数 |
| Learning Rate | 1e-3 | AdamW 优化器 |
| Weight Decay | 1e-4 | L2 正则化 |
| Warmup Epochs | 5 | denoise_cifar_loss 预热轮数 |
| noise_sigma_range | (0.05, 0.25) | CIFAR 噪声增强范围 |
| grad_scale_cls2unet | 0.1 | 分类→UNet 梯度缩放 |
| lambda_denoise_cifar | 1.0 | CIFAR 去噪损失权重 |
| CIFAR batch_size | 64 | |
| SIDD batch_size | 4 | SIDD patch=512×512 |
### 2.4 损失函数
- **分类**：CrossEntropyLoss
- **去噪**：Charbonnier Loss（L1 的平滑近似）：$\mathcal{L} = \sqrt{(pred - target)^2 + \epsilon^2}$，$\epsilon = 10^{-3}$
---
## 3. 消融实验设计
设计了 **5 个模型** 进行消融，逐步分解每个组件的贡献：

| 模型 | 描述 | CIFAR 噪声 | UNet 注入 | SIDD 训练 |
|------|------|:----------:|:---------:|:---------:|
| [A] `standalone_cls` | 干净 CIFAR 训练 ResNet | ✗ | ✗ | ✗ |
| [B] `standalone_cls_noisy` | 噪声 CIFAR 训练 ResNet | ✓ | ✗ | ✗ |
| [C] `multitask_no_sidd` | 噪声 CIFAR + UNet 注入，无 SIDD | ✓ | ✓ | ✗ |
| [D] `multitask_no_cifar_noise` | 干净 CIFAR + UNet 注入 + SIDD | ✗ | ✓ | ✓ |
| [M] `multitask` | 噪声 CIFAR + UNet 注入 + SIDD（完整） | ✓ | ✓ | ✓ |
---
## 4. 实验结果
### 4.1 单任务基线
| 任务 | 指标 | 结果 |
|------|------|------|
| CIFAR-10 分类 [A] | Clean 准确率 | ~84% |
| SIDD 去噪 | PSNR | ~37 dB |
### 4.2 CIFAR-10-C 鲁棒性（五方消融）
| 模型 | CIFAR-10-C 平均准确率 | 相对 A 的提升 |
|------|:--------------------:|:------------:|
| [A] 干净 CIFAR | 60.0% | — |
| [B] 噪声 CIFAR | 61.0% | +1.0% |
| [C] 噪声 + 注入（无 SIDD） | 70.6% | +10.6% |
| [D] 干净 + 注入 + SIDD | 61.1% | +1.1% |
| [M] 噪声 + 注入 + SIDD | **75.0%** | **+15.0%** |
### 4.3 高斯噪声子项（severity 1→3→5）
| 模型 | Severity 1 | Severity 3 | Severity 5 |
|------|:----------:|:----------:|:----------:|
| [A] | ~60% | ~30% | ~20% |
| [B] | ~70% | ~70% | ~70% |
| [M] | ~85% | ~85% | ~85% |
多任务模型[M]在高斯噪声下几乎不受严重等级影响，表现出噪声鲁棒性。
### 4.4 贡献分解
从A(60%)到M(75%)的15%提升来自于：
```
A (60%) ─── +1% ──→ B (61%) ─── +9.6% ──→ C (70.6%) ─── +4.4% ──→ M (75%)
             │                    │                       │
          噪声增强              UNet架构注入              SIDD贡献
```
| 组件 | 贡献 | 占比 |
|------|:----:|:----:|
| 噪声增强（B - A） | +1.0% | 6.7% |
| UNet 架构注入（C - B） | +9.6% | 64.0% |
| SIDD 真实噪声数据（M - C） | +4.4% | 29.3% |
**交叉验证**：
```
A (60%) ─── +1.1% ──→ D (61.1%)     SIDD训练的UNet注入干净CIFAR几乎无用
A (60%) ─── +15% ───→ M (75%)       噪声CIFAR+SIDD：大幅提升
D (61.1%) ─ +13.9% ─→ M (75%)       加上CIFAR噪声后SIDD发力了
```
---
## 5. 一些发现
### 5.1 鲁棒性提升的主要来源
| 发现 | 证据 |
|------|------|
| **UNet 必须在噪声 CIFAR 上训练**才能提供有用特征 | C=70.6% vs D=61.1%（差 9.5%） |
| **噪声增强本身几乎无效** | B=61% vs A=60%（仅+1%） |
| **SIDD 单独无法跨域迁移** | D=61.1% vs A=60%（仅+1.1%） |
| **SIDD 在 UNet 已适应 CIFAR 后提供额外增益** | M=75% vs C=70.6%（+4.4%） |
**核心结论**：鲁棒性提升来自 **UNet 在目标域（噪声 CIFAR）上的联合训练**，而非来自SIDD的知识迁移。SIDD的作用是作为噪声多样性的数据增强补充，而非直接的知识来源。
### 5.2 门控机制无实际作用
v2 的门控值在50epoch训练后几乎未变化：
```
初始化: sigmoid(-2) = 0.119
Epoch 50: L1=0.124, L2=0.105, L3=0.103
变化量:       +0.005     -0.014     -0.016
```
---
## 6. 实验结论
2. **鲁棒性提升的多数来自UNet架构注入，小部分来自SIDD数据**。噪声增强本身仅贡献6.7%。最大贡献者是UNet在噪声 CIFAR上提取的特征。
3. **SIDD的迁移需要在CIFAR噪声训练才有效**。SIDD单独注入干净CIFAR（模型 D）几乎无效（+1.1%），但在UNet已适应CIFAR噪声后，SIDD提供额外+4.4%的增益。
5. **去噪可以促进分类**：UNet PSNR几乎未因分类任务而下降（梯度缩放 0.1× 保护），同时分类鲁棒性大幅提升。