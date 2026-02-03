import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

#预处理流水线 - 训练集使用数据增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 轻微颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#测试集不使用数据增强
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    batch_size=64,
    shuffle=True,
    num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)
print(f"训练集大小: {len(train_dataset)} 张图片")
print(f"测试集大小: {len(test_dataset)} 张图片")

#残差块
#主路径
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
#捷径
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#定义卷积神经网络模型
class Resnet(nn.Module):
    def __init__(self,block,num_classes=10):
        super(Resnet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = block(16, 16, stride=1)
        self.layer2 = block(16, 32, stride=2)
        self.layer3 = block(32, 64, stride=2)
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

#设置设备
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
device = get_device()
print(f"Using device: {device}")

# 创建输出目录
def create_output_dir(base_dir=None):
    # 使用用户主目录下的持久化路径，避免临时目录数据丢失
    if base_dir is None:
        home_dir = os.path.expanduser('~')
        base_dir = os.path.join(home_dir, 'cifar10_training_results')

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
def train(model,train_loader,criterion,optimizer,epochs):
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

# 绘制训练和测试准确率曲线
def plot_training_history(history, save_path='training_history.png'):
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
    # 保存训练历史数据到 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history data saved to {save_path}')


# 主训练流程
if __name__ == '__main__':
    # 创建输出目录
    output_dir = create_output_dir()
    # 创建模型
    model = Resnet(ResidualBlock, num_classes=10).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 训练参数
    epochs = 50
    learning_rate = 0.001
    weight_decay = 0.01
    best_acc = 0

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # 训练循环
    print(f'\n开始训练，共 {epochs} 个epoch...')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epochs)

        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

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
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型，测试准确率: {best_acc:.2f}%')

    # 保存训练历史
    history_json_path = os.path.join(output_dir, 'training_history.json')
    save_training_history(history, history_json_path)

    # 绘制训练曲线
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)

    print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')
    print(f'所有结果已保存到: {output_dir}')

