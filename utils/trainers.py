import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import calculate_psnr, calculate_ssim
from models.multitask import scale_gradient
from datasets.cifar10c import CIFAR10CDataset

# 单任务训练函数

def train_standalone_classifier(model, train_loader, criterion, optimizer, device,
                                noise_sigma_range=None):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    desc = '  [CLS+Noise Train]' if noise_sigma_range else '  [CLS Train]'
    pbar = tqdm(train_loader, desc=desc)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 噪声增强
        if noise_sigma_range is not None:
            sigma = random.uniform(*noise_sigma_range)
            images = torch.clamp(images + torch.randn_like(images) * sigma, 0, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})

    return total_loss / len(train_loader), 100. * correct / total


def train_standalone_denoiser(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    count = 0

    pbar = tqdm(train_loader, desc='  [DEN Train]')
    for noisy, gt in pbar:
        noisy, gt = noisy.to(device), gt.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(noisy)
        loss = criterion(output, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            batch_psnr = calculate_psnr(torch.clamp(output, 0, 1), gt)
            total_psnr += batch_psnr
        count += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'PSNR': f'{batch_psnr:.2f}dB'})

    return total_loss / count, total_psnr / count

# 多任务训练函数

def train_multitask_epoch(model, cifar_loader, sidd_loader, cls_criterion,
                          denoise_criterion, optimizer_main, optimizer_unet,
                          device, epoch, warmup_epochs=5, skip_sidd=False,
                          skip_cifar_noise=False):
    #多任务训练一个 epoch
    model.train()
    warmup = min(1.0, (epoch + 1) / warmup_epochs)

    # 阶段 1: 分类 + CIFAR 去噪 (噪声增强 + 特征注入)
    total_cls, total_dcifar = 0, 0
    cls_correct, cls_total = 0, 0
    total_cifar_psnr = 0
    count1 = 0

    pbar1 = tqdm(cifar_loader, desc=f'  [E{epoch+1} CLS+DEN/CIFAR]')
    for cifar_clean, cifar_labels in pbar1:
        cifar_clean = cifar_clean.to(device)
        cifar_labels = cifar_labels.to(device)

        # 添加随机高斯噪声
        sigma = random.uniform(*model.noise_sigma_range)
        noise = torch.randn_like(cifar_clean) * sigma
        cifar_noisy = torch.clamp(cifar_clean + noise, 0, 1)

        optimizer_main.zero_grad(set_to_none=True)
        optimizer_unet.zero_grad(set_to_none=True)

        # 跨任务前向: UNet去噪 + 特征注入 → ResNet分类
        if skip_cifar_noise:
            with torch.no_grad():
                denoised_tmp, unet_feats_tmp = model.unet(cifar_noisy, return_features=True)
            unet_feats_detached = [f.detach() for f in unet_feats_tmp]
            injections = model._compute_injections(unet_feats_detached)
            logits = model.resnet(cifar_noisy, injections=injections)
            cls_loss = cls_criterion(logits, cifar_labels)
            denoise_cifar_loss = denoise_criterion(denoised_tmp.detach(), cifar_clean)
            result = {
                'cls_loss': cls_loss,
                'denoise_cifar_loss': denoise_cifar_loss,
                'logits': logits,
                'denoised_cifar': denoised_tmp.detach(),
            }
            loss = cls_loss
        else:
            result = model.forward_classify_full(
                cifar_clean, cifar_noisy, cifar_labels,
                cls_criterion, denoise_criterion
            )
            loss = result['cls_loss'] + warmup * model.lambda_denoise_cifar * result['denoise_cifar_loss']

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.resnet.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), max_norm=1.0)
        optimizer_main.step()

        if not skip_cifar_noise:
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
            optimizer_unet.step()

        # 统计
        total_cls += result['cls_loss'].item()
        total_dcifar += result['denoise_cifar_loss'].item()
        _, pred = result['logits'].max(1)
        cls_total += cifar_labels.size(0)
        cls_correct += pred.eq(cifar_labels).sum().item()
        with torch.no_grad():
            cp = calculate_psnr(torch.clamp(result['denoised_cifar'], 0, 1), cifar_clean)
            total_cifar_psnr += cp
        count1 += 1

        pbar1.set_postfix({
            'c': f'{result["cls_loss"].item():.3f}',
            'dc': f'{result["denoise_cifar_loss"].item():.4f}',
            'a': f'{100.*cls_correct/cls_total:.1f}%',
        })

    # 阶段 2: SIDD 去噪
    total_dsidd = 0
    total_sidd_psnr = 0
    count2 = 0

    if skip_sidd:
        return {
            'cls_loss': total_cls / max(count1, 1),
            'denoise_cifar_loss': total_dcifar / max(count1, 1),
            'denoise_sidd_loss': 0.0,
            'cls_acc': 100. * cls_correct / max(cls_total, 1),
            'cifar_psnr': total_cifar_psnr / max(count1, 1),
            'sidd_psnr': 0.0,
        }

    pbar2 = tqdm(sidd_loader, desc=f'  [E{epoch+1} DENOISE/SIDD]')
    for sidd_noisy, sidd_gt in pbar2:
        sidd_noisy = sidd_noisy.to(device)
        sidd_gt = sidd_gt.to(device)

        optimizer_unet.zero_grad(set_to_none=True)

        denoised = model.unet(sidd_noisy)
        loss = denoise_criterion(denoised, sidd_gt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
        optimizer_unet.step()

        total_dsidd += loss.item()
        with torch.no_grad():
            sp = calculate_psnr(torch.clamp(denoised, 0, 1), sidd_gt)
            total_sidd_psnr += sp
        count2 += 1

        pbar2.set_postfix({
            'ds': f'{loss.item():.4f}',
            'P': f'{sp:.1f}dB',
        })

    return {
        'cls_loss': total_cls / max(count1, 1),
        'denoise_cifar_loss': total_dcifar / max(count1, 1),
        'denoise_sidd_loss': total_dsidd / max(count2, 1),
        'cls_acc': 100. * cls_correct / max(cls_total, 1),
        'cifar_psnr': total_cifar_psnr / max(count1, 1),
        'sidd_psnr': total_sidd_psnr / max(count2, 1),
    }

# 评估函数

def evaluate_classifier(model, loader, criterion, device):
    # 评估分类器
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  [CLS Eval]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate_denoiser(model, loader, criterion, device):
    # 评估去噪器
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for noisy, gt in tqdm(loader, desc='  [DEN Eval]'):
            noisy, gt = noisy.to(device), gt.to(device)
            output = model(noisy)
            output = torch.clamp(output, 0, 1)
            loss = criterion(output, gt)
            total_loss += loss.item()
            total_psnr += calculate_psnr(output, gt)
            total_ssim += calculate_ssim(output, gt)
            count += 1

    return total_loss / count, total_psnr / count, total_ssim / count


def evaluate_cifar10c(model, data_dir, device, severities=None):
    # 评估模型在 CIFAR-10-C 上的鲁棒性
    if severities is None:
        severities = [1, 3, 5]

    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'gaussian_blur', 'saturate', 'spatter', 'speckle_noise',
    ]

    transform = transforms.Compose([transforms.ToTensor()])
    results = {}
    model.eval()

    for corruption in corruption_types:
        corruption_file = os.path.join(data_dir, f'{corruption}.npy')
        if not os.path.exists(corruption_file):
            continue

        results[corruption] = {}
        for severity in severities:
            dataset = CIFAR10CDataset(data_dir, corruption, severity, transform)
            loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=4)

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = 100.0 * correct / total
            results[corruption][severity] = acc

        avg_acc = np.mean([results[corruption][s] for s in severities])
        print(f'  {corruption:<25s} 平均={avg_acc:.2f}%  '
              f'[{", ".join(f"S{s}={results[corruption][s]:.2f}" for s in severities)}]')

    return results


def print_cifar10c_comparison(results_standalone, results_multitask, save_path=None):
    # 打印并保存 CIFAR-10-C 鲁棒性对比表
    lines = []
    header = f"{'损坏类型':<25s} {'单独训练':>10s} {'多任务':>10s} {'提升':>10s}"
    lines.append("=" * 60)
    lines.append("  CIFAR-10-C 鲁棒性对比 (平均准确率%)")
    lines.append("=" * 60)
    lines.append(header)
    lines.append("-" * 60)

    all_standalone = []
    all_multitask = []

    all_corruptions = sorted(set(list(results_standalone.keys()) + list(results_multitask.keys())))

    for corruption in all_corruptions:
        if corruption not in results_standalone or corruption not in results_multitask:
            continue
        avg_s = np.mean(list(results_standalone[corruption].values()))
        avg_m = np.mean(list(results_multitask[corruption].values()))
        delta = avg_m - avg_s
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        marker = " ✓" if delta > 0 else ""
        lines.append(f"{corruption:<25s} {avg_s:>10.2f} {avg_m:>10.2f} {delta_str:>10s}{marker}")
        all_standalone.append(avg_s)
        all_multitask.append(avg_m)

    lines.append("-" * 60)
    if all_standalone:
        overall_s = np.mean(all_standalone)
        overall_m = np.mean(all_multitask)
        overall_d = overall_m - overall_s
        delta_str = f"+{overall_d:.2f}" if overall_d >= 0 else f"{overall_d:.2f}"
        lines.append(f"{'总体平均':<25s} {overall_s:>10.2f} {overall_m:>10.2f} {delta_str:>10s}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"鲁棒性对比报告已保存到: {save_path}")

    return report
