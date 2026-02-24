import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import os

from configs.default_config import get_default_config
from models.resnet import ResNetWithFeatures
from models.unet import UNetWithFeatures
from models.multitask import MultiTaskSoftSharingModel
from models.losses import CharbonnierLoss
from datasets.sidd import SIDDDataset
from utils.common import set_seed, get_device, create_output_dir
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import (
    save_history, save_denoising_samples,
    plot_multitask_history, plot_standalone_history,
    plot_cifar10c_comparison, plot_cifar10c_results,
)
from utils.trainers import (
    train_standalone_classifier, train_standalone_denoiser,
    train_multitask_epoch,
    evaluate_classifier, evaluate_denoiser, evaluate_cifar10c,
    print_cifar10c_comparison,
)

# 主训练流程

if __name__ == '__main__':

    # 配置
    config = get_default_config()

    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")

    mode = config['mode']
    print(f"\n{'='*60}")
    print(f"  多任务跨任务特征注入学习框架")
    print(f"  模式: {mode}")
    print(f"{'='*60}\n")

    # CIFAR-10 数据

    transform_train_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test_cifar = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar_train = torchvision.datasets.CIFAR10(
        root=config['cifar_data_dir'], train=True, download=True, transform=transform_train_cifar)
    cifar_test = torchvision.datasets.CIFAR10(
        root=config['cifar_data_dir'], train=False, download=True, transform=transform_test_cifar)

    cifar_train_loader = DataLoader(
        cifar_train, batch_size=config['cifar_batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    cifar_test_loader = DataLoader(
        cifar_test, batch_size=config['cifar_batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    print(f"CIFAR-10 训练集: {len(cifar_train)} 张 | 测试集: {len(cifar_test)} 张")

    # SIDD 数据
    sidd_train_loader = None
    sidd_test_loader = None

    if mode in ['standalone_denoise', 'multitask', 'multitask_no_cifar_noise']:
        print(f'\n加载 SIDD 数据集...')
        sidd_full = SIDDDataset(config['sidd_data_dir'],
                                patch_size=config['sidd_patch_size'], augment=True)
        total_pairs = len(sidd_full)
        test_size = max(1, int(total_pairs * 0.1))
        train_size = total_pairs - test_size

        indices = list(range(total_pairs))
        random.shuffle(indices)
        sidd_train_ds = torch.utils.data.Subset(sidd_full, indices[:train_size])

        sidd_test_full = SIDDDataset(config['sidd_data_dir'],
                                     patch_size=config['sidd_patch_size'], augment=False)
        sidd_test_ds = torch.utils.data.Subset(sidd_test_full, indices[train_size:])

        sidd_train_loader = DataLoader(
            sidd_train_ds, batch_size=config['sidd_batch_size'],
            shuffle=True, num_workers=config['num_workers'],
            pin_memory=True, drop_last=True)
        sidd_test_loader = DataLoader(
            sidd_test_ds, batch_size=config['sidd_batch_size'],
            shuffle=False, num_workers=config['num_workers'], pin_memory=True)

        print(f"SIDD 训练集: {len(sidd_train_ds)} 对 | 测试集: {len(sidd_test_ds)} 对")

    # ==================== 模式 1/1b: 单独训练分类器 ====================
    if mode in ['standalone_cls', 'standalone_cls_noisy']:
        is_noisy = (mode == 'standalone_cls_noisy')
        noise_range = config['noise_sigma_range'] if is_noisy else None
        tag = 'standalone_cls_noisy' if is_noisy else 'standalone_cls'
        output_dir = create_output_dir(f'./multitask_training_results/{tag}', prefix=tag)

        model = ResNetWithFeatures(num_classes=10).to(device)
        print(f"ResNet 参数量: {sum(p.numel() for p in model.parameters()):,}")
        if is_noisy:
            print(f"噪声增强: σ ∈ [{noise_range[0]}, {noise_range[1]}] (Baseline B 消融实验)")
        else:
            print(f"无噪声增强 (Baseline A)")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
                   'noise_augmented': is_noisy}
        best_acc = 0

        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}  (lr: {optimizer.param_groups[0]['lr']:.2e})")
            train_loss, train_acc = train_standalone_classifier(
                model, cifar_train_loader, criterion, optimizer, device,
                noise_sigma_range=noise_range)
            test_loss, test_acc = evaluate_classifier(
                model, cifar_test_loader, criterion, device)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f'  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%')
            print(f'  Test:  loss={test_loss:.4f}, acc={test_acc:.2f}%')

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_resnet.pth'))
                print(f'  ✓ 保存最佳模型: {best_acc:.2f}%')

        # CIFAR-10-C 评估
        print(f"\n{'='*40}")
        print(f"  CIFAR-10-C 鲁棒性评估 ({tag})")
        print(f"{'='*40}")
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_resnet.pth'),
                                         map_location=device, weights_only=True))
        c10c_results = evaluate_cifar10c(model, config['cifar10c_dir'], device,
                                         config['cifar10c_severities'])

        history['best_acc'] = best_acc
        history['cifar10c'] = {k: {str(sk): sv for sk, sv in v.items()}
                               for k, v in c10c_results.items()}
        save_history(history, os.path.join(output_dir, 'history.json'))
        plot_standalone_history(history, os.path.join(output_dir, 'training_curves.png'), task='cls')
        if c10c_results:
            plot_cifar10c_results(c10c_results,
                                 os.path.join(output_dir, 'cifar10c_robustness.png'),
                                 title=f'CIFAR-10-C Robustness ({tag})')
        print(f"\n完成! 最佳准确率: {best_acc:.2f}%  结果保存于: {output_dir}")

    # ==================== 模式 2: 单独训练去噪器 ====================
    elif mode == 'standalone_denoise':
        output_dir = create_output_dir('./multitask_training_results/standalone_denoise',
                                       prefix='Denoise')
        model = UNetWithFeatures(base_channels=config['unet_base_channels'],
                                 num_cab=config['unet_num_cab']).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"UNet 参数量: {total_params:,} ({total_params/1e6:.2f}M)")

        criterion = CharbonnierLoss(eps=1e-3)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                betas=(0.9, 0.9), weight_decay=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-7)

        history = {'train_loss': [], 'train_psnr': [],
                   'test_loss': [], 'test_psnr': [], 'test_ssim': []}
        best_psnr = 0

        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}  (lr: {optimizer.param_groups[0]['lr']:.2e})")
            train_loss, train_psnr = train_standalone_denoiser(
                model, sidd_train_loader, criterion, optimizer, device)
            test_loss, test_psnr, test_ssim = evaluate_denoiser(
                model, sidd_test_loader, criterion, device)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_psnr'].append(train_psnr)
            history['test_loss'].append(test_loss)
            history['test_psnr'].append(test_psnr)
            history['test_ssim'].append(test_ssim)

            print(f'  Train: loss={train_loss:.4f}, PSNR={train_psnr:.2f}dB')
            print(f'  Test:  loss={test_loss:.4f}, PSNR={test_psnr:.2f}dB, SSIM={test_ssim:.4f}')

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_unet.pth'))
                print(f'  ✓ 保存最佳模型: PSNR={best_psnr:.2f}dB')

        history['best_psnr'] = best_psnr
        save_history(history, os.path.join(output_dir, 'history.json'))
        plot_standalone_history(history, os.path.join(output_dir, 'training_curves.png'), task='denoise')
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_unet.pth'),
                                         map_location=device, weights_only=True))
        save_denoising_samples(model, sidd_test_loader, device,
                               os.path.join(output_dir, 'denoising_samples.png'),
                               num_samples=min(4, config['sidd_batch_size']))
        print(f"\n完成! 最佳 PSNR: {best_psnr:.2f}dB  结果保存于: {output_dir}")

    # ==================== 模式 3/3b/3c: 多任务跨任务特征注入训练 ====================
    elif mode in ['multitask', 'multitask_no_sidd', 'multitask_no_cifar_noise']:
        skip_sidd = (mode == 'multitask_no_sidd')
        skip_cifar_noise = (mode == 'multitask_no_cifar_noise')
        tag = mode
        output_dir = create_output_dir(f'./multitask_training_results/{tag}', prefix='MultiTask')

        # 干净CIFAR模式: 将噪声范围设为(0,0)
        actual_noise_range = (0.0, 0.0) if skip_cifar_noise else config['noise_sigma_range']

        model = MultiTaskSoftSharingModel(
            num_classes=10,
            unet_base_ch=config['unet_base_channels'],
            unet_num_cab=config['unet_num_cab'],
            noise_sigma_range=actual_noise_range,
            grad_scale_cls2unet=config['grad_scale_cls2unet'],
            lambda_denoise_cifar=config['lambda_denoise_cifar'],
        ).to(device)

        resnet_params = sum(p.numel() for p in model.resnet.parameters())
        unet_params = sum(p.numel() for p in model.unet.parameters())
        bridge_params = sum(p.numel() for p in model.bridges.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量:")
        print(f"  ResNet:     {resnet_params:>10,}")
        print(f"  UNet:       {unet_params:>10,} ({unet_params/1e6:.2f}M)")
        print(f"  注入桥:     {bridge_params:>10,}")
        print(f"  总计:       {total_params:>10,} ({total_params/1e6:.2f}M)")
        print(f"\n跨任务参数:")
        print(f"  噪声范围:   σ ∈ [{actual_noise_range[0]}, {actual_noise_range[1]}]")
        print(f"  梯度缩放:   cls→UNet = {config['grad_scale_cls2unet']}")
        print(f"  CIFAR去噪权重: λ = {config['lambda_denoise_cifar']}")
        if skip_sidd:
            print(f"  SIDD 训练已禁用 (Baseline C: 消融 SIDD 贡献)")
        if skip_cifar_noise:
            print(f"  CIFAR 噪声已禁用 (Baseline D: 干净CIFAR + SIDD注入)")

        cls_criterion = nn.CrossEntropyLoss()
        denoise_criterion = CharbonnierLoss(eps=1e-3)

        optimizer_main = optim.AdamW(
            list(model.resnet.parameters()) + list(model.bridges.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay'])
        optimizer_unet = optim.AdamW(
            model.unet.parameters(),
            lr=config['learning_rate'], betas=(0.9, 0.9), weight_decay=0)

        scheduler_main = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_main, T_max=config['epochs'], eta_min=1e-7)
        scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_unet, T_max=config['epochs'], eta_min=1e-7)

        history = {
            'config': {k: str(v) if isinstance(v, tuple) else v for k, v in config.items()},
            'train_cls_loss': [], 'train_cls_acc': [],
            'test_cls_loss': [], 'test_cls_acc': [],
            'train_denoise_cifar_loss': [], 'train_cifar_psnr': [],
            'train_denoise_sidd_loss': [], 'train_sidd_psnr': [],
            'test_denoise_loss': [], 'test_psnr': [], 'test_ssim': [],
            'lr': [],
        }

        best_cls_acc = 0
        best_psnr = 0

        #计算 SIDD noisy 基线
        if not skip_sidd:
            print('\n  [REF] 计算 SIDD noisy 基线...')
            _noisy_psnr_total, _noisy_ssim_total, _noisy_count = 0, 0, 0
            with torch.no_grad():
                for _noisy, _gt in sidd_test_loader:
                    _noisy, _gt = _noisy.to(device), _gt.to(device)
                    _noisy_psnr_total += calculate_psnr(_noisy, _gt)
                    _noisy_ssim_total += calculate_ssim(_noisy, _gt)
                    _noisy_count += 1
            _noisy_baseline_psnr = _noisy_psnr_total / _noisy_count
            _noisy_baseline_ssim = _noisy_ssim_total / _noisy_count
            print(f'  [REF] Noisy baseline: PSNR={_noisy_baseline_psnr:.2f}dB, '
                  f'SSIM={_noisy_baseline_ssim:.4f}')

        # 训练循环
        print(f'\n===== 开始{tag}训练，共 {config["epochs"]} 个 epoch =====\n')
        for epoch in range(config['epochs']):
            current_lr = optimizer_main.param_groups[0]['lr']
            print(f'\nEpoch {epoch+1}/{config["epochs"]}  (lr: {current_lr:.2e})')

            train_metrics = train_multitask_epoch(
                model, cifar_train_loader, sidd_train_loader,
                cls_criterion, denoise_criterion,
                optimizer_main, optimizer_unet, device,
                epoch, warmup_epochs=config['warmup_epochs'],
                skip_sidd=skip_sidd, skip_cifar_noise=skip_cifar_noise)

            test_cls_loss, test_cls_acc = evaluate_classifier(
                model, cifar_test_loader, cls_criterion, device)

            if not skip_sidd:
                test_den_loss, test_psnr, test_ssim = evaluate_denoiser(
                    model.unet, sidd_test_loader, denoise_criterion, device)
            else:
                test_den_loss, test_psnr, test_ssim = 0.0, 0.0, 0.0

            scheduler_main.step()
            scheduler_unet.step()

            # 记录
            history['train_cls_loss'].append(train_metrics['cls_loss'])
            history['train_cls_acc'].append(train_metrics['cls_acc'])
            history['test_cls_loss'].append(test_cls_loss)
            history['test_cls_acc'].append(test_cls_acc)
            history['train_denoise_cifar_loss'].append(train_metrics['denoise_cifar_loss'])
            history['train_cifar_psnr'].append(train_metrics['cifar_psnr'])
            history['train_denoise_sidd_loss'].append(train_metrics['denoise_sidd_loss'])
            history['train_sidd_psnr'].append(train_metrics['sidd_psnr'])
            history['test_denoise_loss'].append(test_den_loss)
            history['test_psnr'].append(test_psnr)
            history['test_ssim'].append(test_ssim)
            history['lr'].append(current_lr)

            print(f'  [CLS]   Train acc={train_metrics["cls_acc"]:.2f}%  |  '
                  f'Test acc={test_cls_acc:.2f}%')
            if not skip_sidd:
                print(f'  [DEN]   CIFAR PSNR={train_metrics["cifar_psnr"]:.2f}dB  |  '
                      f'SIDD Train PSNR={train_metrics["sidd_psnr"]:.2f}dB  |  '
                      f'SIDD Test PSNR={test_psnr:.2f}dB, SSIM={test_ssim:.4f}')
            else:
                print(f'  [DEN]   CIFAR PSNR={train_metrics["cifar_psnr"]:.2f}dB  |  '
                      f'SIDD: 已禁用')

            # 打印门控值
            gate_vals = []
            for i, bridge in enumerate(model.bridges):
                bias = bridge.gate[2].bias.data.mean().item()
                gate_avg = torch.sigmoid(torch.tensor(bias)).item()
                gate_vals.append(f'L{i+1}={gate_avg:.3f}')
            print(f'  [GATE]  {", ".join(gate_vals)}  (注入强度, 近1=强注入)')

            if test_cls_acc > best_cls_acc:
                best_cls_acc = test_cls_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_multitask_cls.pth'))
                print(f'  ✓ 保存最佳分类模型: {best_cls_acc:.2f}%')

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_multitask_denoise.pth'))
                print(f'  ✓ 保存最佳去噪模型: PSNR={best_psnr:.2f}dB')

        # 评估
        history['best_cls_acc'] = best_cls_acc
        history['best_psnr'] = best_psnr

        print(f"\n{'='*60}")
        print(f"  CIFAR-10-C 鲁棒性评估 ({tag})")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_multitask_cls.pth'),
                                         map_location=device, weights_only=True))
        c10c_results = evaluate_cifar10c(
            model, config['cifar10c_dir'], device, config['cifar10c_severities'])

        history['cifar10c'] = {k: {str(sk): sv for sk, sv in v.items()}
                               for k, v in c10c_results.items()}

        save_history(history, os.path.join(output_dir, 'history.json'))
        plot_multitask_history(history, os.path.join(output_dir, 'training_curves.png'))

        if c10c_results:
            plot_cifar10c_results(c10c_results,
                                 os.path.join(output_dir, 'cifar10c_robustness.png'),
                                 title=f'CIFAR-10-C Robustness ({tag})')

        if not skip_sidd:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'best_multitask_denoise.pth'),
                                             map_location=device, weights_only=True))
            save_denoising_samples(model.unet, sidd_test_loader, device,
                                   os.path.join(output_dir, 'denoising_samples.png'),
                                   num_samples=min(4, config['sidd_batch_size']))

        print(f"\n===== {tag} 训练完成！=====")
        print(f"最佳分类准确率: {best_cls_acc:.2f}%")
        if not skip_sidd:
            print(f"最佳去噪 PSNR: {best_psnr:.2f}dB")
        print(f"结果保存于: {output_dir}")

    else:
        print(f"未知模式: {mode}")
        print("可用模式: standalone_cls, standalone_cls_noisy, standalone_denoise, "
              "multitask_no_sidd, multitask_no_cifar_noise, multitask")
