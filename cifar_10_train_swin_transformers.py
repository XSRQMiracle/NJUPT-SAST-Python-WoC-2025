import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

#预处理流水线 - 训练集使用数据增强
transform_train = transforms.Compose([
    transforms.Resize(224),  #将32×32 resize到224×224
    transforms.RandomCrop(224, padding=28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 标准化参数
])

#测试集不使用数据增强
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR-10数据集
data_path = './cifar10_data'

train_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=True,
    download=True,
    transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=False,
    download=True,
    transform=transform_test
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)
print(f"训练集大小: {len(train_dataset)} 张图片")
print(f"测试集大小: {len(test_dataset)} 张图片")
#Swin Transformer 模型定义
class SwinTransformer_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, model_name='swin_tiny_patch4_window7_224'):
        super(SwinTransformer_CIFAR10, self).__init__()

        # 加载预训练的 Swin Transformer 模型
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


#设备选择
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")
#创建输出目录
def create_output_dir(base_dir=None):
    if base_dir is None:
        base_dir = './cifar10_training_results_swin'

    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    existing_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))
                     and d.startswith('第') and 'CIFAR-10' in d]
    max_num = 0
    for dir_name in existing_dirs:
        try:
            num_str = dir_name.split('次')[0].replace('第', '')
            num = int(num_str)
            max_num = max(max_num, num)
        except:
            continue

    new_num = max_num + 1
    output_dir = os.path.join(base_dir, f'第{new_num}次CIFAR-10_{date_str}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'输出目录: {output_dir}')
    print(f'完整路径: {os.path.abspath(output_dir)}')
    return output_dir


# 训练
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(train_loader), 100. * correct / total


#评估
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


#绘图
def plot_training_history(history, save_path='training_history.png'):
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['test_acc'], label='Test Acc', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Swin Transformer - Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Swin Transformer - Training and Test Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Training history plot saved to {save_path}')


# 保存训练历史
def save_training_history(history, save_path='training_history.json'):
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history data saved to {save_path}')


# 主训练流程
if __name__ == '__main__':
    # 创建输出目录
    output_dir = create_output_dir()

    # 模型配置
    model_name = 'swin_tiny_patch4_window7_224'  # 使用 Swin-Tiny
    pretrained = True  # 使用预训练权重

    # 创建模型
    model = SwinTransformer_CIFAR10(
        num_classes=10,
        pretrained=pretrained,
        model_name=model_name
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {model_name}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 训练参数
    epochs = 30
    learning_rate = 1e-4
    weight_decay = 0.05  # Swin Transformer 推荐的权重衰减
    batch_size = 64
    best_acc = 0
    best_loss = float('inf')
    best_epoch = 0

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 使用 AdamW 优化器（Swin Transformer 推荐）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # 记录超参数
    hyperparameters = {
        'model': model_name,
        'pretrained': pretrained,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'optimizer': 'AdamW',
        'scheduler': f'CosineAnnealingLR (T_max={epochs}, eta_min=1e-6)',
        'loss_function': 'CrossEntropyLoss',
        'input_size': '224x224 (resized from 32x32)',
        'data_augmentation': 'Resize(224), RandomCrop(224, padding=28), RandomHorizontalFlip',
        'normalization': 'mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(device)
    }

    # 训练历史记录
    history = {
        'hyperparameters': hyperparameters,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rates': []
    }

    # 训练循环
    print(f'\n开始训练 Swin Transformer，共 {epochs} 个 epoch...')

    for epoch in range(epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f'\nEpoch {epoch + 1}/{epochs} (lr: {current_lr:.2e})')

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(current_lr)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_loss = test_loss
            best_epoch = epoch + 1
            model_path = os.path.join(output_dir, 'best_swin_cifar10.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'hyperparameters': hyperparameters
            }, model_path)
            print(f'✓ 保存最佳模型，测试准确率: {best_acc:.2f}%')

    # 添加最佳结果到历史记录
    history['best_test_acc'] = best_acc
    history['best_test_loss'] = best_loss
    history['best_epoch'] = best_epoch

    # 保存训练历史
    history_json_path = os.path.join(output_dir, 'training_history.json')
    save_training_history(history, history_json_path)

    # 绘制训练曲线
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)

    # 打印训练总结
    print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'所有结果已保存到: {output_dir}')
