import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualBlock(nn.Module):
    # 残差块
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetWithFeatures(nn.Module):
    # ResNet主干网络, 支持返回中间层特征和接受跨任务注入
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = ResidualBlock(16, 16, stride=1)    # [B, 16, 32, 32]
        self.layer2 = ResidualBlock(16, 32, stride=2)    # [B, 32, 16, 16]
        self.layer3 = ResidualBlock(32, 64, stride=2)    # [B, 64,  8,  8]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_features=False, injections=None):
        # 注入机制: 在每个层级输出后加上对应的注入特征
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)     # Level 1: [B, 16, H, W]
        if injections is not None and injections[0] is not None:
            f1 = f1 + injections[0]
        f2 = self.layer2(f1)      # Level 2: [B, 32, H/2, W/2]
        if injections is not None and injections[1] is not None:
            f2 = f2 + injections[1]
        f3 = self.layer3(f2)      # Level 3: [B, 64, H/4, W/4]
        if injections is not None and injections[2] is not None:
            f3 = f3 + injections[2]

        pooled = self.avg_pool(f3)
        logits = self.fc(torch.flatten(pooled, 1))

        if return_features:
            return logits, [f1, f2, f3]
        return logits

# ResNet-18模型（单任务训练用）
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18_CIFAR10, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        # 修改conv1以适应32*32输入
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 移除 maxpool 层
        self.model.maxpool = nn.Identity()

        # 修改最后的全连接层
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

# 手搓的简化版 ResNet（单任务训练用）
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = ResidualBlock(16, 16, stride=1)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
