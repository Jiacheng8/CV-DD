import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Remember to change me!
data_dir = ""  # where the dataset are stored
save_dir = ""  # where to save the trained models

# Overall configuration
num_classes = 10
batch_size = 64
num_epochs = 300
model_lis = ['resnet18', 'resnet50', 'densenet121', 'shufflenetV2', 'mobilenetv2']
pretrain=False

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def load_model(model_name):
    if model_name=='resnet18':
        model = models.resnet18(pretrained=pretrain)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name=='resnet50':
        model = models.resnet50(pretrained=pretrain)
        model.fc = nn.Linear(model.fc.in_features, num_classes) 
    elif model_name=='densenet121':
        model = models.densenet121(pretrained=pretrain)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    elif model_name=='shufflenetV2':
        model = models.shufflenet_v2_x1_0(pretrained=pretrain)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name=='mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrain)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Invalid model name")
    return model

def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 每 10 个 batch 打印一次当前的损失和准确率
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 每 10 个 batch 打印一次验证损失和准确率
            if batch_idx % 10 == 0:
                print(f"Validation Batch [{batch_idx}/{len(loader)}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


for model_name in model_lis:
    criterion = nn.CrossEntropyLoss()
    model = load_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,                # base learning rate
        momentum=0.9,          # optimizer momentum
        weight_decay=1e-4      # weight decay
    )

    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=100
    )

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Completed")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")
        print("-" * 50)

    # 保存微调后的模型
    torch.save(model.state_dict(), f"{save_dir}/{model_name}.pth")
    print(f"Finished processing {model_name}")