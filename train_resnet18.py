import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from models.resnet import ResNet18_CIFAR10
from utils.common import get_device, create_output_dir
from utils.visualization import plot_training_history, save_history

# 训练集使用数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 测试集不使用数据增强
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# 训练和评估函数
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


# 主训练流程

if __name__ == '__main__':
    # 设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载数据集
    data_path = './cifar10_data'
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=8)

    print(f"训练集大小: {len(train_dataset)} 张图片")
    print(f"测试集大小: {len(test_dataset)} 张图片")

    # 创建输出目录
    output_dir = create_output_dir(
        base_dir='./cifar10_training_results_resnet18', prefix='CIFAR-10')

    # 创建模型
    model = ResNet18_CIFAR10(num_classes=10).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 训练参数
    epochs = 100
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    best_acc = 0
    best_loss = float('inf')
    best_epoch = 0

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)

    # 记录超参数
    hyperparameters = {
        'model': 'ResNet-18',
        'epochs': epochs,
        'learning_rate': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'batch_size': 128,
        'optimizer': 'SGD',
        'scheduler': 'MultiStepLR (milestones=[30, 60, 80], gamma=0.2)',
        'loss_function': 'CrossEntropyLoss',
        'data_augmentation': 'RandomCrop(32, padding=4), RandomHorizontalFlip',
        'normalization': 'mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)',
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device)
    }

    # 训练历史
    history = {
        'hyperparameters': hyperparameters,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # 训练循环
    print(f'\n开始训练，共 {epochs} 个epoch...')
    for epoch in range(epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f'\nEpoch {epoch + 1}/{epochs} (lr: {current_lr:.6f})')

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_loss = test_loss
            best_epoch = epoch + 1
            model_path = os.path.join(output_dir, 'best_resnet18_cifar10.pth')
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型，测试准确率: {best_acc:.2f}%')

    history['best_test_acc'] = best_acc
    history['best_test_loss'] = best_loss
    history['best_epoch'] = best_epoch

    save_history(history, os.path.join(output_dir, 'training_history.json'))
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))

    print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'所有结果已保存到: {output_dir}')
