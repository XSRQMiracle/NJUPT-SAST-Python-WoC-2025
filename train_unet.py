import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import os

from models.unet import UNet
from models.losses import CharbonnierLoss
from datasets.sidd import SIDDDataset
from utils.common import set_seed, get_device, create_output_dir
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import (
    plot_standalone_history, save_history, save_denoising_samples
)
from utils.trainers import train_standalone_denoiser, evaluate_denoiser


if __name__ == '__main__':
    set_seed(1234)

    device = get_device()
    print(f"Using device: {device}")

    # 超参数
    data_dir = './SIDD_Medium_Srgb/Data'
    patch_size = 512
    batch_size = 2
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 0
    num_workers = 8
    test_ratio = 0.1
    base_channels = 64

    # 创建输出目录
    output_dir = create_output_dir(
        base_dir='./sidd_training_results_unet', prefix='SIDD')

    # ===== 构建数据集 =====
    print('\n===== 加载 SIDD 数据集 =====')
    full_dataset = SIDDDataset(data_dir, patch_size=patch_size, augment=True)

    total_pairs = len(full_dataset)
    test_size = max(1, int(total_pairs * test_ratio))
    train_size = total_pairs - test_size

    indices = list(range(total_pairs))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

    # 测试集不做数据增强
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

    # 损失函数 & 优化器
    criterion = CharbonnierLoss(eps=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.9),
                            weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # 超参数记录
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

        train_loss, train_psnr = train_standalone_denoiser(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_psnr, test_ssim = evaluate_denoiser(
            model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_psnr'].append(train_psnr)
        history['test_loss'].append(test_loss)
        history['test_psnr'].append(test_psnr)
        history['test_ssim'].append(test_ssim)
        history['lr'].append(current_lr)

        print(f'Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}dB')
        print(f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}dB, Test SSIM: {test_ssim:.4f}')

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_ssim = test_ssim
            best_epoch = epoch + 1
            model_path = os.path.join(output_dir, 'best_unet_sidd.pth')
            torch.save(model.state_dict(), model_path)
            print(f'✓ 保存最佳模型，Test PSNR: {best_psnr:.2f}dB, Test SSIM: {best_ssim:.4f}')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            sample_path = os.path.join(output_dir, f'denoise_samples_epoch{epoch + 1}.png')
            save_denoising_samples(model, test_loader, device, sample_path)

    history['best_test_psnr'] = best_psnr
    history['best_test_ssim'] = best_ssim
    history['best_epoch'] = best_epoch

    save_history(history, os.path.join(output_dir, 'training_history.json'))
    plot_standalone_history(history, os.path.join(output_dir, 'training_history.png'), task='denoise')

    final_sample_path = os.path.join(output_dir, 'final_denoise_samples.png')
    save_denoising_samples(model, test_loader, device, final_sample_path)

    print(f'\n===== 训练完成！=====')
    print(f'最佳 Test PSNR: {best_psnr:.2f}dB (Epoch {best_epoch})')
    print(f'最佳 Test SSIM: {best_ssim:.4f}')
    print(f'所有结果已保存到: {output_dir}')
