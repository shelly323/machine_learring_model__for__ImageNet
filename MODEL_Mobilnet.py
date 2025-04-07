# 建立模型
import torch
import torchvision as tv
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader  # 修正：dataloader -> DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

# 繪製圖形
import matplotlib.pyplot as plt
import numpy as np

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料處理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # MobileNetV1/V2 输入尺寸
    transforms.RandomHorizontalFlip(),  # 建议添加数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # 添加中心裁剪以匹配训练尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 載入資料集
train_dataset = datasets.ImageFolder(root="D:/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="D:/val", transform=val_transform)

train_loader = DataLoader(  # 修正：dataloader -> DataLoader
    train_dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=4  # 建议使用多线程加载数据
)
val_loader = DataLoader(  # 修正：dataloader -> DataLoader
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4  # 建议使用多线程加载数据
)

# Mobilenet_v2
model = models.mobilenet_v2(pretrained=True)
num_classes = len(train_dataset.classes)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# loss function
criterion = torch.nn.CrossEntropyLoss()

# optimizer - 只优化分类层
optimizer = optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 訓練參數
epochs = 20
best_val_acc = 0.0

# 训练和验证循环
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(epochs):
    # 训练阶段
    model.train()  # 修正：model.train -> model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # 修正：input -> inputs
        _, preds = torch.max(outputs, 1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.cpu().numpy())  # 转换为numpy便于绘图
    
    # 验证阶段
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
    
    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset)
    
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc.cpu().numpy())  # 转换为numpy便于绘图
    
    # 更新学习率
    scheduler.step()
    
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
    
    # 保存最佳模型
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), 'best_model.pth')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()