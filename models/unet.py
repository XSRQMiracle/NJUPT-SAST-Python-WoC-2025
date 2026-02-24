import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    # 通道注意力 (SE-like)
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class CAB(nn.Module):
    # 通道注意力残差块 (Channel Attention Block)
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.ca = ChannelAttention(channels, reduction=4)

    def forward(self, x):
        return self.ca(self.body(x)) + x


class Downsample(nn.Module):
    # 下采样模块
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    # 标准 UNet 模型 (用于图像去噪)
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_cab=2):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # 浅层特征提取
        self.shallow_feat = nn.Conv2d(in_channels, c1, 3, padding=1, bias=True)

        # 编码器
        self.enc1 = nn.Sequential(*[CAB(c1) for _ in range(num_cab)])
        self.down1 = Downsample(c1, c2)

        self.enc2 = nn.Sequential(*[CAB(c2) for _ in range(num_cab)])
        self.down2 = Downsample(c2, c3)

        self.enc3 = nn.Sequential(*[CAB(c3) for _ in range(num_cab)])
        self.down3 = Downsample(c3, c4)

        self.enc4 = nn.Sequential(*[CAB(c4) for _ in range(num_cab)])
        self.down4 = Downsample(c4, c4)

        # 瓶颈层
        self.bottleneck = nn.Sequential(*[CAB(c4) for _ in range(4)])

        # 解码器
        self.up4 = Upsample(c4, c4)
        self.reduce4 = nn.Conv2d(c4 * 2, c4, 1, bias=True)
        self.dec4 = nn.Sequential(*[CAB(c4) for _ in range(num_cab)])

        self.up3 = Upsample(c4, c3)
        self.reduce3 = nn.Conv2d(c3 * 2, c3, 1, bias=True)
        self.dec3 = nn.Sequential(*[CAB(c3) for _ in range(num_cab)])

        self.up2 = Upsample(c3, c2)
        self.reduce2 = nn.Conv2d(c2 * 2, c2, 1, bias=True)
        self.dec2 = nn.Sequential(*[CAB(c2) for _ in range(num_cab)])

        self.up1 = Upsample(c2, c1)
        self.reduce1 = nn.Conv2d(c1 * 2, c1, 1, bias=True)
        self.dec1 = nn.Sequential(*[CAB(c1) for _ in range(num_cab)])

        # 输出层
        self.tail = nn.Conv2d(c1, out_channels, 3, padding=1, bias=True)

    def forward(self, x):
        # 全局残差学习: output = model_output + noisy_input
        identity = x

        # 浅层特征
        x = self.shallow_feat(x)

        # 编码器
        e1 = self.enc1(x)       # [B, c1,  H,   W]
        x = self.down1(e1)      # [B, c2, H/2, W/2]

        e2 = self.enc2(x)       # [B, c2, H/2, W/2]
        x = self.down2(e2)      # [B, c3, H/4, W/4]

        e3 = self.enc3(x)       # [B, c3, H/4, W/4]
        x = self.down3(e3)      # [B, c4, H/8, W/8]

        e4 = self.enc4(x)       # [B, c4, H/8, W/8]
        x = self.down4(e4)      # [B, c4, H/16, W/16]

        # 瓶颈层
        x = self.bottleneck(x)  # [B, c4, H/16, W/16]

        # 解码器 + Skip Connection
        x = self.up4(x)
        x = self.reduce4(torch.cat([x, e4], dim=1))
        x = self.dec4(x)

        x = self.up3(x)
        x = self.reduce3(torch.cat([x, e3], dim=1))
        x = self.dec3(x)

        x = self.up2(x)
        x = self.reduce2(torch.cat([x, e2], dim=1))
        x = self.dec2(x)

        x = self.up1(x)
        x = self.reduce1(torch.cat([x, e1], dim=1))
        x = self.dec1(x)

        # 输出 + 全局残差连接
        x = self.tail(x)
        x = x + identity

        return x


class UNetWithFeatures(nn.Module):
    # UNet 模型变体, 支持返回编码器中间特征 (用于跨任务注入)
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_cab=2):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # 浅层特征
        self.shallow_feat = nn.Conv2d(in_channels, c1, 3, padding=1, bias=True)

        # 编码器
        self.enc1 = nn.Sequential(*[CAB(c1) for _ in range(num_cab)])
        self.down1 = Downsample(c1, c2)
        self.enc2 = nn.Sequential(*[CAB(c2) for _ in range(num_cab)])
        self.down2 = Downsample(c2, c3)
        self.enc3 = nn.Sequential(*[CAB(c3) for _ in range(num_cab)])
        self.down3 = Downsample(c3, c4)
        self.enc4 = nn.Sequential(*[CAB(c4) for _ in range(num_cab)])
        self.down4 = Downsample(c4, c4)

        # 瓶颈层
        self.bottleneck = nn.Sequential(*[CAB(c4) for _ in range(4)])

        # 解码器
        self.up4 = Upsample(c4, c4)
        self.reduce4 = nn.Conv2d(c4 * 2, c4, 1, bias=True)
        self.dec4 = nn.Sequential(*[CAB(c4) for _ in range(num_cab)])

        self.up3 = Upsample(c4, c3)
        self.reduce3 = nn.Conv2d(c3 * 2, c3, 1, bias=True)
        self.dec3 = nn.Sequential(*[CAB(c3) for _ in range(num_cab)])

        self.up2 = Upsample(c3, c2)
        self.reduce2 = nn.Conv2d(c2 * 2, c2, 1, bias=True)
        self.dec2 = nn.Sequential(*[CAB(c2) for _ in range(num_cab)])

        self.up1 = Upsample(c2, c1)
        self.reduce1 = nn.Conv2d(c1 * 2, c1, 1, bias=True)
        self.dec1 = nn.Sequential(*[CAB(c1) for _ in range(num_cab)])

        # 输出层
        self.tail = nn.Conv2d(c1, out_channels, 3, padding=1, bias=True)

    def forward(self, x, return_features=False):
        identity = x  # 全局残差学习

        x = self.shallow_feat(x)

        # 编码器
        e1 = self.enc1(x)         # [B, c1, H,   W]
        x = self.down1(e1)
        e2 = self.enc2(x)         # [B, c2, H/2, W/2]
        x = self.down2(e2)
        e3 = self.enc3(x)         # [B, c3, H/4, W/4]
        x = self.down3(e3)
        e4 = self.enc4(x)         # [B, c4, H/8, W/8]
        x = self.down4(e4)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码器 + Skip
        x = self.up4(x)
        x = self.reduce4(torch.cat([x, e4], dim=1))
        x = self.dec4(x)
        x = self.up3(x)
        x = self.reduce3(torch.cat([x, e3], dim=1))
        x = self.dec3(x)
        x = self.up2(x)
        x = self.reduce2(torch.cat([x, e2], dim=1))
        x = self.dec2(x)
        x = self.up1(x)
        x = self.reduce1(torch.cat([x, e1], dim=1))
        x = self.dec1(x)

        output = self.tail(x) + identity

        if return_features:
            return output, [e1, e2, e3]
        return output

    def forward_encoder_only(self, x):
        # 前向传播编码器部分, 返回中间特征 (用于跨任务注入)
        x = self.shallow_feat(x)
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.enc2(x)
        x = self.down2(e2)
        e3 = self.enc3(x)
        return [e1, e2, e3]
