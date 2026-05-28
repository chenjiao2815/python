import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *

class RoadDataset(Dataset):
    def __init__(self, transform=None, split='train', val_ratio=0.2):
        self.images_dir = TRAIN_IMAGES_DIR
        self.masks_dir = TRAIN_MASKS_DIR
        self.transform = transform
        
        # 获取所有图像和掩码文件名
        all_images = sorted(os.listdir(self.images_dir))
        all_masks = sorted(os.listdir(self.masks_dir))
        
        # 创建掩码文件名到路径的映射（去掉扩展名）
        mask_map = {os.path.splitext(mask)[0]: mask for mask in all_masks}
        
        # 只保留有对应掩码的图像
        paired_images = []
        paired_masks = []
        
        for img_name in all_images:
            img_base = os.path.splitext(img_name)[0]
            if img_base in mask_map:
                paired_images.append(img_name)
                paired_masks.append(mask_map[img_base])
        
        # 划分训练集和验证集
        import random
        random.seed(42)
        indices = list(range(len(paired_images)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - val_ratio))
        if split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]
        
        self.images = [paired_images[i] for i in selected_indices]
        self.masks = [paired_masks[i] for i in selected_indices]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask

def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]) 