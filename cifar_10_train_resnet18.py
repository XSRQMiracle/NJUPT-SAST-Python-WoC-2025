import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

#预处理流水线 - 训练集使用数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

#测试集不使用数据增强
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

#加载CIFAR-10数据集
data_path = './cifar10_data'

train_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=True,
    download=True,
    transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=False,
    download=True,
    transform=transform_test)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8
)
print(f"训练集大小: {len(train_dataset)} 张图片")
print(f"测试集大小: {len(test_dataset)} 张图片")


# Resnet-18模型，修改
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18_CIFAR10, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        # 修改conv1以32*32输入
        # 原始：kernel_size=7, stride=2, padding=3
        # 修改：kernel_size=3, stride=1, padding=1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 移除 maxpool 层
        self.model.maxpool = nn.Identity()

        # 修改最后的全连接层改为分成10类
        # nn.Linear(512, 1000) -> nn.Linear(512, 10)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


#评估模型
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

#选择设备
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def plot_training_history(history, save_path='training_history.png'):
    #绘制训练和测试准确率曲线
    plt.figure(figsize=(12, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['test_acc'], label='Test Acc', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 损失曲线
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
    print(f'Training history plot saved to {save_path}')


def save_training_history(history, save_path='training_history.json'):
    #保存训练历史数据到 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history data saved to {save_path}')


#设置设备
device = get_device()
print(f"Using device: {device}")

# 创建输出目录
def create_output_dir(base_dir=None):
    # 保存到脚本所在目录
    if base_dir is None:
        base_dir = './cifar10_training_results_resnet18'

    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('第') and 'CIFAR-10' in d]
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

#训练
def train(model, train_loader, criterion, optimizer, epochs):
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


# 主训练流程
if __name__ == '__main__':
    # 创建输出目录
    output_dir = create_output_dir()

    # 创建模型
    model = ResNet18_CIFAR10(num_classes=10).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 训练参数
    epochs = 100
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 128
    best_acc = 0
    best_loss = float('inf')
    best_epoch = 0

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)

    # 记录超参数
    hyperparameters = {
        'model': 'ResNet-18',
        'epochs': epochs,
        'learning_rate': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'optimizer': 'SGD',
        'scheduler': 'MultiStepLR (milestones=[30, 60, 80], gamma=0.2)',
        'loss_function': 'CrossEntropyLoss',
        'data_augmentation': 'RandomCrop(32, padding=4), RandomHorizontalFlip',
        'normalization': 'mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)',
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device)
    }

    # 训练历史记录
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

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epochs)

        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_loss = test_loss
            best_epoch = epoch + 1
            model_path = os.path.join(output_dir, 'best_resnet18_cifar10.pth')
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型，测试准确率: {best_acc:.2f}%')

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

    print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'所有结果已保存到: {output_dir}')
