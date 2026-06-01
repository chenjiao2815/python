import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


def get_model():
    # model = smp.Unet(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     in_channels=3,
    #     classes=1,
    #     activation=ACTIVATION,
    # )
    model = smp.DaDeepLabV3Plus(
        encoder_name=ENCODER,
        in_channels=3,
        classes=1,
        activation=ACTIVATION,
        encoder_weights=None
    )
    return model


class DiceCELoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5, ignore_index=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = smp.losses.DiceLoss(
            mode="binary", from_logits=True, ignore_index=ignore_index
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        """
        logits: shape (B, C, H, W)
        targets: shape (B, H, W)  for multiclass (as label indices)
        """
        if logits.shape[1] == 1:  # binary case, convert targets to same shape
            targets = targets.unsqueeze(1).float()
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets.long())
        return self.dice_weight * dice + self.ce_weight * ce


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits: [B, 1, H, W], targets: [B, H, W]
        if logits.shape[1] != 1:
            raise ValueError(f"Expected logits shape [B,1,H,W], got {logits.shape}")
        targets = targets.unsqueeze(1).float()  # [B,1,H,W]
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce

class RoadLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss替代CrossEntropyLoss
        self.alpha = alpha  # 连续性权重系数

    def continuity_loss(self, pred):
        # 计算横向/纵向梯度差异
        dx = pred[:,:,:,1:] - pred[:,:,:,:-1]
        dy = pred[:,:,:,1:] - pred[:,:,:,:-1]
        return torch.mean(torch.exp(-(dx**2 + dy**2)))  # 惩罚剧烈变化

    def forward(self, pred, target):
        # 确保维度匹配
        if target.dim() == 3:
            target = target.unsqueeze(1)
        bce_loss = self.bce(pred, target)
        cont_loss = self.continuity_loss(pred)
        return bce_loss + self.alpha * cont_loss


def get_loss():
    if LOSS == 'dice':
        return smp.losses.DiceLoss(mode='binary')
    elif LOSS == 'dicebce':
        return DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
    elif LOSS == 'bce':
        class SoftBCEWithLogitsLossWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = smp.losses.SoftBCEWithLogitsLoss()

            def forward(self, outputs, targets):
                # 确保维度匹配
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                return self.loss(outputs, targets)

        return SoftBCEWithLogitsLossWrapper()
    elif LOSS == 'jaccard':
        return smp.losses.JaccardLoss(mode='binary')
    elif LOSS == 'focal':
        return smp.losses.FocalLoss(mode='binary', gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    elif LOSS == 'lovasz':
        return smp.losses.LovaszLoss(mode='binary')
    elif LOSS == 'tversky':
        return smp.losses.TverskyLoss(mode='binary', alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
    elif LOSS == 'road':
        return RoadLoss()
    elif LOSS == 'combo':
        # 组合损失函数
        dice_loss = smp.losses.DiceLoss()
        bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        focal_loss = smp.losses.FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

        class ComboLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice_loss = dice_loss
                self.bce_loss = bce_loss
                self.focal_loss = focal_loss

            def forward(self, outputs, targets):
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                return (LOSS_WEIGHTS['dice'] * self.dice_loss(outputs, targets) +
                        LOSS_WEIGHTS['bce'] * self.bce_loss(outputs, targets) +
                        LOSS_WEIGHTS['focal'] * self.focal_loss(outputs, targets))

        return ComboLoss()
    else:
        raise ValueError(f"Unknown loss type: {LOSS}")


def calculate_metrics(outputs, masks, threshold=0.5):
    outputs = (outputs > threshold).float()
    masks = masks.float()

    # 确保维度匹配
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    tp = torch.sum(outputs * masks)
    fp = torch.sum(outputs * (1 - masks))
    fn = torch.sum((1 - outputs) * masks)
    tn = torch.sum((1 - outputs) * (1 - masks))

    iou = smp.metrics.iou_score(tp, fp, fn, tn)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn)
    recall = smp.metrics.recall(tp, fp, fn, tn)
    precision = smp.metrics.precision(tp, fp, fn, tn)

    return {
        'iou': iou.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item(),
        'recall': recall.item(),
        'precision': precision.item()
    }


def get_metrics():
    return [
        smp.metrics.IoU(threshold=0.5),
        smp.metrics.Fscore(threshold=0.5),
    ]
