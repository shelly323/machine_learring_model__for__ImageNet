import torch
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
import numpy as np

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料前處理 (包含資料增強)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 載入資料集
train_dataset = datasets.ImageFolder(root="D:/train", transform=train_transform)

val_dataset = datasets.ImageFolder(root="D:/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #num_workers=4
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
    #num_workers=4
)

# RESNET50
model = models.resnet50(pretrained=True)
#獲取數據集的數量
num_classes = len(train_dataset.classes)

# 凍結所有層
for param in model.parameters():
    param.requires_grad = False

# 替換最後的全連接層（適應自定義類別數）
model.fc = torch.nn.Linear(
    model.fc.in_features, 
    num_classes
)
#用指定的裝置
model = model.to(device)

# 損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
# 只優化最後一層
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)  
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 學習率調整

# 訓練參數
epochs = 20
best_val_acc = 0.0

# 訓練迴圈
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    train_losses = []
    train_correct = []

    # 訓練階段
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        #獲取類別，忽略value
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    scheduler.step()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    train_losses.append(epoch_loss)
    train_correct.append(epoch_acc.cpu().numpy())
    
    # 驗證階段
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    val_losses = []
    val_Corrects = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    val_losses.append(val_loss)
    val_Corrects.append(val_acc.cpu().numpy())
    
    # 打印結果
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model saved!')

print(f'Best Val Acc: {best_val_acc:.4f}')



#繪製圖型

plt.figure(figsize=(12,5))

#Loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, labels = 'Train Loss')
plt.plot(train_correct, labels = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_correct, labels = 'Train Accuracy')
plt.plot(val_Corrects, labels = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()