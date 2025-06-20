import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        preds_sigmoid = torch.sigmoid(preds)
        smooth = 1.0
        intersection = (preds_sigmoid * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (preds_sigmoid.sum() + targets.sum() + smooth)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss