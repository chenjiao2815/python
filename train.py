import os
# 👇 在这里加一句：禁用 albumentations 联网检查（解决你的网络报错）
os.environ["ALBUMENTATIONS_UPDATE_CHECK"] = "0"

import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from dataset import RoadDataset, get_train_transform, get_val_transform
from model import get_model, get_loss, calculate_metrics
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    torch.backends.cudnn.deterministic = True  # 保证卷积结果可复现
    torch.backends.cudnn.benchmark = False     # 关闭自动优化卷积算法

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {
        'iou': 0,
        'f1': 0,
        'accuracy': 0,
        'recall': 0,
        'precision': 0
    }

    example_images = []
    example_masks = []
    example_outputs = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            metrics = calculate_metrics(outputs, masks)
            for key in metrics:
                total_metrics[key] += metrics[key]

            if batch_idx == 0:
                example_images = images[:4].cpu()
                example_masks = masks[:4].cpu()
                example_outputs = outputs[:4].cpu()

    avg_metrics = {key: value / len(loader) for key, value in total_metrics.items()}
    return total_loss / len(loader), avg_metrics

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = RoadDataset(transform=get_train_transform())
    val_dataset = RoadDataset(transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # 👇 改成 0，Windows 最稳定，不报错
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0  # 👇 同样改成 0
    )

    model = get_model().to(device)
    criterion = get_loss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Train Loss: {train_loss:.4f}')
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
        print(val_metrics)
        print(f'Val Loss: {val_loss:.4f}')
        for metric_name, metric_value in val_metrics.items():
            print(f'Val {metric_name}: {metric_value:.4f}')

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print('Model saved!')

# 👇 你原来这里是对的，我保留不动
if __name__ == '__main__':
    main()