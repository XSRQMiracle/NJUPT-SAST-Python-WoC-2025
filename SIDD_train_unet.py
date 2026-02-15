import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import glob
import random
import numpy as np
from datetime import datetime
from PIL import Image, ImageFile
from math import log10



# 随机数种子
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(1234)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# SIDD数据集
class SIDDDataset(Dataset):
    def __init__(self, data_dir, patch_size=512, augment=True, cache_in_memory=False):
        super().__init__()
        self.patch_size = patch_size
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.image_cache = {}

        # 收集所有配对
        self.pairs = []
        scene_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))

        for scene_dir in scene_dirs:
            if not os.path.isdir(scene_dir):
                continue
            gt_files = sorted(glob.glob(os.path.join(scene_dir, '*_GT_SRGB_*.PNG')))
            for gt_path in gt_files:
                noisy_path = gt_path.replace('_GT_SRGB_', '_NOISY_SRGB_')
                if os.path.exists(noisy_path):
                    self.pairs.append((noisy_path, gt_path))

        print(f"找到 {len(self.pairs)} 对 NOISY-GT 图像配对")

    def _load_image(self, path):
        if self.cache_in_memory and path in self.image_cache:
            return self.image_cache[path]

        img = Image.open(path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0

        if self.cache_in_memory:
            self.image_cache[path] = img
        return img
# resize到512*512
    def _resize(self, noisy, gt):
        ps = self.patch_size
        noisy = np.array(Image.fromarray((noisy * 255).astype(np.uint8)).resize((ps, ps), Image.BICUBIC), dtype=np.float32) / 255.0
        gt = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize((ps, ps), Image.BICUBIC), dtype=np.float32) / 255.0
        return noisy, gt
# 数据增强
    def _augment(self, noisy, gt):
        # 随机水平翻转
        if random.random() > 0.5:
            noisy = np.flip(noisy, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
        # 随机垂直翻转
        if random.random() > 0.5:
            noisy = np.flip(noisy, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()
        # 随机旋转 90°
        k = random.randint(0, 3)
        if k > 0:
            noisy = np.rot90(noisy, k).copy()
            gt = np.rot90(gt, k).copy()
        return noisy, gt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            noisy_path, gt_path = self.pairs[idx]
            noisy = self._load_image(noisy_path)
            gt = self._load_image(gt_path)
        except (OSError, IOError, SyntaxError) as e:
            # 图片损坏时随机选另一张
            print(f"\n⚠ 跳过损坏图片: {self.pairs[idx][0]}, 错误: {e}")
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

        noisy, gt = self._resize(noisy, gt)

        if self.augment:
            noisy, gt = self._augment(noisy, gt)

        # 转为tensor
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()

        return noisy, gt


# ===================== 通道注意力块 (CAB) =====================
class ChannelAttention(nn.Module):
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
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn


class CAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.ca = ChannelAttention(channels, reduction=4)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return res + x


# 下采样和上采样模块
class Downsample(nn.Module):
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

# UNet模型
class UNet(nn.Module):
    """带通道注意力的 UNet 去噪模型
    架构: 4级编码器-解码器 + Skip Connection + 全局残差学习
    通道数: 64 → 128 → 256 → 512
    每级使用 2 个 CAB (Channel Attention Block)
    不使用 BatchNorm (去噪任务中 BN 会破坏像素精度)
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_cab=2):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        # 64, 128, 256, 512

        # 浅层特征提取
        self.shallow_feat = nn.Conv2d(in_channels, c1, 3, padding=1, bias=True)

        # ===== 编码器 =====
        self.enc1 = nn.Sequential(*[CAB(c1) for _ in range(num_cab)])
        self.down1 = Downsample(c1, c2)

        self.enc2 = nn.Sequential(*[CAB(c2) for _ in range(num_cab)])
        self.down2 = Downsample(c2, c3)

        self.enc3 = nn.Sequential(*[CAB(c3) for _ in range(num_cab)])
        self.down3 = Downsample(c3, c4)

        self.enc4 = nn.Sequential(*[CAB(c4) for _ in range(num_cab)])
        self.down4 = Downsample(c4, c4)

        # ===== 瓶颈层 =====
        self.bottleneck = nn.Sequential(*[CAB(c4) for _ in range(4)])

        # ===== 解码器 =====
        self.up4 = Upsample(c4, c4)
        self.reduce4 = nn.Conv2d(c4 * 2, c4, 1, bias=True)  # skip concat 后降通道
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
        #全局残差学习: output = model_output + noisy_input
        identity = x  # 保存输入用于全局残差
        # 浅层特征
        x = self.shallow_feat(x)

        # 编码器
        e1 = self.enc1(x)       # [B, 64,  H,   W]
        x = self.down1(e1)      # [B, 128, H/2, W/2]

        e2 = self.enc2(x)       # [B, 128, H/2, W/2]
        x = self.down2(e2)      # [B, 256, H/4, W/4]

        e3 = self.enc3(x)       # [B, 256, H/4, W/4]
        x = self.down3(e3)      # [B, 512, H/8, W/8]

        e4 = self.enc4(x)       # [B, 512, H/8, W/8]
        x = self.down4(e4)      # [B, 512, H/16, W/16]

        # 瓶颈层
        x = self.bottleneck(x)  # [B, 512, H/16, W/16]

        # 解码器 + Skip Connection
        x = self.up4(x)                         # [B, 512, H/8, W/8]
        x = self.reduce4(torch.cat([x, e4], dim=1))
        x = self.dec4(x)

        x = self.up3(x)                         # [B, 256, H/4, W/4]
        x = self.reduce3(torch.cat([x, e3], dim=1))
        x = self.dec3(x)

        x = self.up2(x)                         # [B, 128, H/2, W/2]
        x = self.reduce2(torch.cat([x, e2], dim=1))
        x = self.dec2(x)

        x = self.up1(x)                         # [B, 64,  H,   W]
        x = self.reduce1(torch.cat([x, e1], dim=1))
        x = self.dec1(x)

        # 输出 + 全局残差连接
        x = self.tail(x)
        x = x + identity

        return x


# CharbonnierLoss损失函数
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps2))
        return loss


# 评估
def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * log10(1.0 / mse)


def calculate_ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    channel = pred.size(1)

    # 创建高斯窗口
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)

    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(pred.device).type_as(pred)

    # 计算均值
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()

# 计算 SSIM Map
def calculate_ssim_map(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    channel = pred.size(1)

    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)

    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(pred.device).type_as(pred)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 取通道平均，返回 [H, W] numpy
    return ssim_map.squeeze(0).mean(dim=0).cpu().numpy()


# 选择设备
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")


# 输出目录
def create_output_dir(base_dir=None):
    if base_dir is None:
        base_dir = './sidd_training_results_unet'

    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    existing_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))
                     and d.startswith('第') and 'SIDD' in d]
    max_num = 0
    for dir_name in existing_dirs:
        try:
            num_str = dir_name.split('次')[0].replace('第', '')
            num = int(num_str)
            max_num = max(max_num, num)
        except:
            continue
    new_num = max_num + 1
    output_dir = os.path.join(base_dir, f'第{new_num}次SIDD_{date_str}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录: {output_dir}')
    print(f'完整路径: {os.path.abspath(output_dir)}')
    return output_dir


# 训练
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    count = 0

    pbar = tqdm(train_loader, desc='Training')
    for noisy, gt in pbar:
        noisy, gt = noisy.to(device), gt.to(device)

        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        output = model(noisy)
        loss = criterion(output, gt)  # 训练时不 clamp，保持梯度畅通
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 计算训练 PSNR (此时再 clamp 到 [0,1])
        with torch.no_grad():
            batch_psnr = calculate_psnr(torch.clamp(output, 0, 1), gt)
            total_psnr += batch_psnr

        count += 1
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'PSNR': f'{batch_psnr:.2f}dB'
        })

    avg_loss = total_loss / count
    avg_psnr = total_psnr / count
    return avg_loss, avg_psnr


# 评估
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for noisy, gt in tqdm(test_loader, desc='Evaluating'):
            noisy, gt = noisy.to(device), gt.to(device)

            output = model(noisy)
            output = torch.clamp(output, 0, 1)
            loss = criterion(output, gt)

            total_loss += loss.item()
            total_psnr += calculate_psnr(output, gt)
            total_ssim += calculate_ssim(output, gt)
            count += 1

    avg_loss = total_loss / count
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_loss, avg_psnr, avg_ssim


# 可视化
def plot_training_history(history, save_path='training_history.png'):
    plt.figure(figsize=(18, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # PSNR 曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_psnr'], label='Train PSNR', color='blue')
    plt.plot(history['test_psnr'], label='Test PSNR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Training and Test PSNR')
    plt.legend()
    plt.grid(True)

    # SSIM 曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['test_ssim'], label='Test SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Test SSIM')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存到 {save_path}')

# 保存去噪效果对比图
def save_denoising_samples(model, test_loader, device, save_path, num_samples=4):
    model.eval()
    noisy_batch, gt_batch = next(iter(test_loader))
    noisy_batch, gt_batch = noisy_batch.to(device), gt_batch.to(device)

    with torch.no_grad():
        output_batch = model(noisy_batch)
        output_batch = torch.clamp(output_batch, 0, 1)

    num_samples = min(num_samples, noisy_batch.size(0))
    fig, axes = plt.subplots(num_samples, 6, figsize=(30, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        noisy_img = noisy_batch[i].cpu().permute(1, 2, 0).numpy()
        output_img = output_batch[i].cpu().permute(1, 2, 0).numpy()
        gt_img = gt_batch[i].cpu().permute(1, 2, 0).numpy()

        psnr_noisy = calculate_psnr(noisy_batch[i:i+1], gt_batch[i:i+1])
        psnr_denoised = calculate_psnr(output_batch[i:i+1], gt_batch[i:i+1])

        # 残差图（放大5倍）
        residual = np.abs(output_img - gt_img)
        residual_vis = np.clip(residual * 5.0, 0, 1)

        # 方法噪声图（归一化）
        method_noise = noisy_img - output_img
        method_noise_vis = np.clip(method_noise * 2.0 + 0.5, 0, 1)

        # SSIM Map
        with torch.no_grad():
            ssim_map = calculate_ssim_map(output_batch[i:i+1], gt_batch[i:i+1])

        # 第1列: Noisy
        axes[i, 0].imshow(np.clip(noisy_img, 0, 1))
        axes[i, 0].set_title(f'Noisy\nPSNR: {psnr_noisy:.2f}dB')
        axes[i, 0].axis('off')

        # 第2列: Denoised
        axes[i, 1].imshow(np.clip(output_img, 0, 1))
        axes[i, 1].set_title(f'Denoised\nPSNR: {psnr_denoised:.2f}dB')
        axes[i, 1].axis('off')

        # 第3列: GT
        axes[i, 2].imshow(np.clip(gt_img, 0, 1))
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

        # 第4列: 残差图 |Denoised - GT| × 5
        axes[i, 3].imshow(residual_vis)
        axes[i, 3].set_title(f'Residual |D-GT|×5\nMAE: {residual.mean():.4f}')
        axes[i, 3].axis('off')

        # 第5列: 方法噪声图 (Noisy - Denoised)
        axes[i, 4].imshow(method_noise_vis)
        axes[i, 4].set_title('Method Noise\n(Noisy - Denoised)')
        axes[i, 4].axis('off')

        # 第6列: SSIM Map
        im = axes[i, 5].imshow(ssim_map, cmap='jet', vmin=0, vmax=1)
        axes[i, 5].set_title(f'SSIM Map\nMean: {ssim_map.mean():.4f}')
        axes[i, 5].axis('off')
        plt.colorbar(im, ax=axes[i, 5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'去噪效果对比图已保存到 {save_path}')


def save_training_history(history, save_path='training_history.json'):
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'训练历史已保存到 {save_path}')


# 主训练流程
if __name__ == '__main__':
    # 创建输出目录
    output_dir = create_output_dir()

    # 超参数
    data_dir = './SIDD_Medium_Srgb/Data'
    patch_size = 512           # 裁剪尺寸 (RTX 4090 24GB 下 512×512 可用 batch_size=2)
    batch_size = 2             # RTX 4090 + 512×512 + UNet-64ch → batch_size=2
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 0           # 去噪任务一般不用 weight decay
    num_workers = 8
    test_ratio = 0.1            # 10% 的场景用于测试
    base_channels = 64         # UNet 基础通道数

    # ===== 构建数据集 =====
    print('\n===== 加载 SIDD 数据集 =====')
    full_dataset = SIDDDataset(data_dir, patch_size=patch_size, augment=True)

    # 按场景划分训练集/测试集 (确保同一场景不同时出现在训练和测试集中)
    total_pairs = len(full_dataset)
    test_size = max(1, int(total_pairs * test_ratio))
    train_size = total_pairs - test_size

    # 固定划分
    indices = list(range(total_pairs))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # 测试集不做数据增强 (通过创建一个不增强的新 dataset)
    test_dataset_no_aug = SIDDDataset(data_dir, patch_size=patch_size, augment=False)
    test_dataset = torch.utils.data.Subset(test_dataset_no_aug, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"训练集大小: {len(train_dataset)} 对图像")
    print(f"测试集大小: {len(test_dataset)} 对图像")

    # 创建模型
    model = UNet(in_channels=3, out_channels=3, base_channels=base_channels, num_cab=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 损失函数
    criterion = CharbonnierLoss(eps=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.9),
                            weight_decay=weight_decay)
    # 学习率调度器：Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # 超参数
    hyperparameters = {
        'model': 'UNet with Channel Attention (CAB)',
        'base_channels': base_channels,
        'num_cab_per_level': 2,
        'levels': 4,
        'global_residual': True,
        'patch_size': patch_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW (betas=0.9, 0.9)',
        'scheduler': f'CosineAnnealingLR (T_max={epochs}, eta_min=1e-7)',
        'loss_function': 'CharbonnierLoss (eps=1e-3)',
        'data_augmentation': 'RandomHFlip + RandomVFlip + RandomRot90',
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'model_parameters': total_params,
        'device': str(device)
    }

    # 训练历史
    history = {
        'hyperparameters': hyperparameters,
        'train_loss': [],
        'train_psnr': [],
        'test_loss': [],
        'test_psnr': [],
        'test_ssim': [],
        'lr': []
    }

    # 训练循环
    best_psnr = 0
    best_ssim = 0
    best_epoch = 0

    print(f'\n===== 开始训练，共 {epochs} 个 epoch =====')
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{epochs}  (lr: {current_lr:.2e})')

        # 训练
        train_loss, train_psnr = train(model, train_loader, criterion, optimizer, device)

        # 测试
        test_loss, test_psnr, test_ssim = evaluate(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_psnr'].append(train_psnr)
        history['test_loss'].append(test_loss)
        history['test_psnr'].append(test_psnr)
        history['test_ssim'].append(test_ssim)
        history['lr'].append(current_lr)

        print(f'Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}dB')
        print(f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}dB, Test SSIM: {test_ssim:.4f}')

        # 保存最佳模型
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_ssim = test_ssim
            best_epoch = epoch + 1
            model_path = os.path.join(output_dir, 'best_unet_sidd.pth')
            torch.save(model.state_dict(), model_path)
            print(f'✓ 保存最佳模型，Test PSNR: {best_psnr:.2f}dB, Test SSIM: {best_ssim:.4f}')

        # 10个epoch保存一次去噪效果对比图
        if (epoch + 1) % 10 == 0 or epoch == 0:
            sample_path = os.path.join(output_dir, f'denoise_samples_epoch{epoch + 1}.png')
            save_denoising_samples(model, test_loader, device, sample_path)

    history['best_test_psnr'] = best_psnr
    history['best_test_ssim'] = best_ssim
    history['best_epoch'] = best_epoch

    # 保存训练历史JSON
    history_json_path = os.path.join(output_dir, 'training_history.json')
    save_training_history(history, history_json_path)

    # 绘制训练曲线
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)

    # 保存最终去噪效果对比
    final_sample_path = os.path.join(output_dir, 'final_denoise_samples.png')
    save_denoising_samples(model, test_loader, device, final_sample_path)

    print(f'\n===== 训练完成！=====')
    print(f'最佳 Test PSNR: {best_psnr:.2f}dB (Epoch {best_epoch})')
    print(f'最佳 Test SSIM: {best_ssim:.4f}')
    print(f'所有结果已保存到: {output_dir}')