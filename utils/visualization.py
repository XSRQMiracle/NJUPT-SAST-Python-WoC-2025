"""可视化与保存工具"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .metrics import calculate_psnr, calculate_ssim_map


def save_history(history, save_path):
    """保存训练历史到 JSON"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f'训练历史已保存到 {save_path}')

# 分类训练曲线 (ResNet 单任务)
def plot_training_history(history, save_path='training_history.png'):
    """绘制分类训练曲线 (loss + accuracy)"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['test_acc'], label='Test Acc', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存到 {save_path}')


# 单任务训练曲线 (分类 / 去噪)
def plot_standalone_history(history, save_path, task='cls'):
    """绘制单任务训练曲线"""
    if task == 'cls':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history['train_loss'], label='Train', color='blue')
        axes[0].plot(history['test_loss'], label='Test', color='red')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_acc'], label='Train', color='blue')
        axes[1].plot(history['test_acc'], label='Test', color='red')
        axes[1].set_title('Accuracy (%)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(history['train_loss'], label='Train', color='blue')
        axes[0].plot(history['test_loss'], label='Test', color='red')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_psnr'], label='Train', color='blue')
        axes[1].plot(history['test_psnr'], label='Test', color='red')
        axes[1].set_title('PSNR (dB)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(history['test_ssim'], label='Test', color='green')
        axes[2].set_title('SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存到: {save_path}')

# 多任务训练曲线 (分类 + 去噪)
def plot_multitask_history(history, save_path):
    """绘制多任务训练曲线 (2×3 子图)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 分类损失
    axes[0, 0].plot(history['train_cls_loss'], label='Train', color='blue')
    axes[0, 0].plot(history['test_cls_loss'], label='Test', color='red')
    axes[0, 0].set_title('Classification Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 分类准确率
    axes[0, 1].plot(history['train_cls_acc'], label='Train', color='blue')
    axes[0, 1].plot(history['test_cls_acc'], label='Test', color='red')
    axes[0, 1].set_title('Classification Accuracy (%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 去噪损失
    axes[0, 2].plot(history['train_denoise_cifar_loss'], label='CIFAR Denoise', color='blue')
    axes[0, 2].plot(history['train_denoise_sidd_loss'], label='SIDD Denoise', color='orange')
    axes[0, 2].plot(history['test_denoise_loss'], label='SIDD Test', color='red')
    axes[0, 2].set_title('Denoising Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # SIDD PSNR
    axes[1, 0].plot(history['train_sidd_psnr'], label='Train', color='blue')
    axes[1, 0].plot(history['test_psnr'], label='Test', color='red')
    axes[1, 0].set_title('SIDD PSNR (dB)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # SSIM
    if history.get('test_ssim'):
        axes[1, 1].plot(history['test_ssim'], label='Test SSIM', color='green')
        axes[1, 1].set_title('SIDD Test SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # CIFAR Denoise PSNR
    axes[1, 2].plot(history['train_cifar_psnr'], label='CIFAR Denoise PSNR', color='purple')
    axes[1, 2].set_title('CIFAR Denoise PSNR (dB)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存到: {save_path}')

# CIFAR-10-C 鲁棒性可视化
def plot_cifar10c_comparison(results_standalone, results_multitask, save_path):
    """绘制 CIFAR-10-C 鲁棒性对比柱状图"""
    corruptions = sorted(set(results_standalone.keys()) & set(results_multitask.keys()))
    if not corruptions:
        print("无共同损坏类型可对比")
        return

    avg_standalone = [np.mean(list(results_standalone[c].values())) for c in corruptions]
    avg_multitask = [np.mean(list(results_multitask[c].values())) for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, avg_standalone, width, label='Standalone ResNet', color='steelblue')
    ax.bar(x + width / 2, avg_multitask, width, label='Multi-Task (Feature Injection)', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CIFAR-10-C Robustness Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, (s, m) in enumerate(zip(avg_standalone, avg_multitask)):
        delta = m - s
        color = 'green' if delta > 0 else 'red'
        ax.annotate(f'{delta:+.1f}', xy=(x[i] + width / 2, m),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=6, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'鲁棒性对比图已保存到: {save_path}')


def plot_cifar10c_results(results, save_path, title='CIFAR-10-C Robustness'):
    """绘制单模型 CIFAR-10-C 鲁棒性柱状图"""
    if not results:
        print("无 CIFAR-10-C 数据可绘图")
        return

    corruptions = sorted(results.keys())
    severities = sorted({s for v in results.values() for s in v.keys()})

    fig, ax = plt.subplots(figsize=(16, 6))
    width = 0.8 / max(len(severities), 1)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(severities)))

    for j, sev in enumerate(severities):
        vals = [results[c].get(sev, 0) for c in corruptions]
        x = np.arange(len(corruptions))
        ax.bar(x + j * width, vals, width, label=f'Severity {sev}', color=colors[j])

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(corruptions)) + width * (len(severities) - 1) / 2)
    ax.set_xticklabels(corruptions, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    overall_avg = np.mean([results[c][s] for c in corruptions for s in results[c]])
    ax.axhline(y=overall_avg, color='navy', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(corruptions) - 0.5, overall_avg + 0.5,
            f'Avg: {overall_avg:.1f}%', color='navy', fontsize=9, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'CIFAR-10-C 鲁棒性图已保存到: {save_path}')

# 去噪效果对比图
def save_denoising_samples(model, test_loader, device, save_path, num_samples=4):
    """
    保存去噪效果对比图 (每行 6 张):
    Noisy | Denoised | GT | 残差图 |D-GT|×5 | 方法噪声 (Noisy-Denoised) | SSIM Map
    """
    import torch

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

        residual = np.abs(output_img - gt_img)
        residual_vis = np.clip(residual * 5.0, 0, 1)

        method_noise = noisy_img - output_img
        method_noise_vis = np.clip(method_noise * 2.0 + 0.5, 0, 1)

        with torch.no_grad():
            ssim_map = calculate_ssim_map(output_batch[i:i+1], gt_batch[i:i+1])

        axes[i, 0].imshow(np.clip(noisy_img, 0, 1))
        axes[i, 0].set_title(f'Noisy\nPSNR: {psnr_noisy:.2f}dB')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(output_img, 0, 1))
        axes[i, 1].set_title(f'Denoised\nPSNR: {psnr_denoised:.2f}dB')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(np.clip(gt_img, 0, 1))
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(residual_vis)
        axes[i, 3].set_title(f'Residual |D-GT|×5\nMAE: {residual.mean():.4f}')
        axes[i, 3].axis('off')

        axes[i, 4].imshow(method_noise_vis)
        axes[i, 4].set_title('Method Noise\n(Noisy - Denoised)')
        axes[i, 4].axis('off')

        im = axes[i, 5].imshow(ssim_map, cmap='jet', vmin=0, vmax=1)
        axes[i, 5].set_title(f'SSIM Map\nMean: {ssim_map.mean():.4f}')
        axes[i, 5].axis('off')
        plt.colorbar(im, ax=axes[i, 5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'去噪效果对比图已保存到: {save_path}')
