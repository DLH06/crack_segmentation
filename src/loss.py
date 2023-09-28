import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = torch.sum(logits * targets)
        union = torch.sum(logits) + torch.sum(targets)
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coefficient


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = self.dice_loss(
            F.softmax(logits, dim=1), F.one_hot(targets, num_classes=logits.shape[1])
        )
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()
        return focal_loss
