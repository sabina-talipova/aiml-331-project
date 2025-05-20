import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from model import VisionTransformer
import time

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=f'runs/vit_heads{config["heads"]}_layers{config["layers"]}_patch{config["patch"]}_pos{config["pos"]}')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(config['data_path'] + '/train.py', transform=transform)
    val_ds   = datasets.ImageFolder(config['data_path'] + '/val', transform=transform)
    test_ds  = datasets.ImageFolder(config['data_path'] + '/test', transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_loader  = DataLoader(test_ds, batch_size=32)

    model = VisionTransformer(
        image_size=128,
        patch_size=config['patch'],
        num_heads=config['heads'],
        num_layers=config['layers'],
        use_positional=config['pos']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        writer.add_scalar('Loss/train.py', train_loss, epoch)
        writer.add_scalar('Accuracy/train.py', train_acc, epoch)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Final test accuracy
    test_correct = 0
    test_total = 0
    model.eval()
    start = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)
    end = time.time()
    test_acc = test_correct / test_total
    print(f'Final Test Accuracy: {test_acc:.4f}, Inference time: {(end-start)/test_total:.6f}s per image')

    writer.close()

