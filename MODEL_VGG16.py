import torch 
import torchvision as tv
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt  # 修正matplotlib導入方式
from tqdm import tqdm  # 進度條工具

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料處理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),  # MobileNetV1/V2 輸入尺寸
    transforms.RandomHorizontalFlip(),  # 數據增強
    transforms.RandomRotation(15),  # 新增旋轉增強
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 載入資料集
train_dataset = datasets.ImageFolder(root="D:/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="D:/val", transform=val_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # 加速數據傳輸到GPU
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 初始化模型
model = models.vgg16(pretrained=True)
num_classes = len(train_dataset.classes)

# 凍結特徵提取層
for param in model.features.parameters():
    param.requires_grad = False

# 修改最後全連接層
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

# 損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 使用AdamW
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 訓練參數
epochs = 50
best_val_acc = 0.0
patience = 5  # early stopping耐心值
no_improve = 0

# 記錄指標
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# 混合精度訓練
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    # 訓練階段
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # 使用tqdm進度條
    train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}')
    for inputs, labels in train_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # 混合精度訓練
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # 更新進度條
        train_bar.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.cpu().numpy())
    
    # 驗證階段
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    val_bar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{epochs}')
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            
            val_bar.set_postfix(loss=loss.item())
    
    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset)
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc.cpu().numpy())
    
    # 更新學習率
    scheduler.step()
    
    print(f'\nEpoch {epoch+1}/{epochs}')
    print(f'Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
    print(f'Val Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}')
    
    # Early stopping 機制
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model!")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

# 繪製結果
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

plt.tight_layout()
plt.savefig('training_results.png')  # 保存圖像
plt.show()