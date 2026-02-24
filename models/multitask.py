import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNetWithFeatures
from .unet import UNetWithFeatures
from .losses import CharbonnierLoss


class _GradientScale(torch.autograd.Function):
   # 用于缩放反向传播时的梯度 (前向不变)
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x, scale):
    # 如果 scale=1.0 则直接返回 x, 否则通过 _GradientScale 缩放梯度
    if scale == 1.0:
        return x
    return _GradientScale.apply(x, scale)


class CrossTaskBridge(nn.Module):
    # 跨任务特征注入桥
    # 将 UNet 编码器特征通过1×1卷积通道对齐+门控加法注入ResNet中间层
    def __init__(self, unet_ch, resnet_ch):
        super().__init__()
        # 1×1 Conv: UNet 通道 → ResNet 通道
        self.transform = nn.Sequential(
            nn.Conv2d(unet_ch, resnet_ch, 1, bias=False),
            nn.BatchNorm2d(resnet_ch),
        )
        # Channel-wise门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(resnet_ch, resnet_ch),
            nn.Sigmoid()
        )
        # 初始化门控偏置 → sigmoid(-2) ≈ 0.12 (保守注入)
        nn.init.constant_(self.gate[2].bias, -2.0)

    def forward(self, unet_feat, target_hw=None):
        # 如果 UNet 特征空间尺寸不匹配 ResNet 目标层, 则先插值调整
        if target_hw is not None and unet_feat.shape[2:] != target_hw:
            unet_feat = F.interpolate(unet_feat, size=target_hw,
                                      mode='bilinear', align_corners=False)
        transformed = self.transform(unet_feat)                    # [B, resnet_ch, H, W]
        gate = self.gate(transformed).unsqueeze(-1).unsqueeze(-1)  # [B, resnet_ch, 1, 1]
        return gate * transformed


class MultiTaskSoftSharingModel(nn.Module):
    # 多任务模型主体, 包含 ResNet 分类器 + UNet 去噪器 + 跨任务桥
    def __init__(self, num_classes=10, unet_base_ch=64, unet_num_cab=2,
                 noise_sigma_range=(0.05, 0.25), grad_scale_cls2unet=0.1,
                 lambda_denoise_cifar=1.0):
        super().__init__()
        self.noise_sigma_range = noise_sigma_range
        self.grad_scale_cls2unet = grad_scale_cls2unet
        self.lambda_denoise_cifar = lambda_denoise_cifar

        # 任务专有网络
        self.resnet = ResNetWithFeatures(num_classes=num_classes)
        self.unet = UNetWithFeatures(base_channels=unet_base_ch, num_cab=unet_num_cab)

        # 跨任务特征注入桥 (3 个层级: UNet encoder → ResNet)
        resnet_chs = [16, 32, 64]
        unet_chs = [unet_base_ch, unet_base_ch * 2, unet_base_ch * 4]
        self.bridges = nn.ModuleList([
            CrossTaskBridge(u_ch, r_ch)
            for u_ch, r_ch in zip(unet_chs, resnet_chs)
        ])

    def _compute_injections(self, unet_feats):
        # 计算跨任务注入信号: 将 UNet 编码器特征通过对应的 Bridge 转换为 ResNet 注入信号
        return [bridge(uf) for bridge, uf in zip(self.bridges, unet_feats)]

    def forward(self, x):
        # 前向推理: 直接在输入图像上进行分类 (不使用 UNet 特征注入)
        unet_feats = self.unet.forward_encoder_only(x)
        injections = self._compute_injections(unet_feats)
        return self.resnet(x, injections=injections)

    def forward_classify_full(self, x_clean, x_noisy, labels,
                              cls_criterion, denoise_criterion):
        # 前向推理: 同时进行分类和去噪, 使用 UNet 特征注入 ResNet
        # UNet: 完整前向 = 去噪 + 编码器特征
        denoised, unet_feats = self.unet(x_noisy, return_features=True)

        # 缩放 cls→UNet 梯度, 防止分类任务主导 UNet 训练
        scaled_feats = [scale_gradient(f, self.grad_scale_cls2unet) for f in unet_feats]

        # Bridge: UNet 特征 → 注入信号
        injections = self._compute_injections(scaled_feats)

        # ResNet: 带注入的分类
        logits = self.resnet(x_noisy, injections=injections)

        # 损失
        cls_loss = cls_criterion(logits, labels)
        denoise_loss = denoise_criterion(denoised, x_clean)

        return {
            'cls_loss': cls_loss,
            'denoise_cifar_loss': denoise_loss,
            'logits': logits,
            'denoised_cifar': denoised,
        }
