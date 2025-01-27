from monai.losses.dice import DiceLoss
import torch.nn as nn
import logging


class SoftDiceLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(SoftDiceLoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.dice = DiceLoss(sigmoid=True)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = self.dice(input, target)
        return loss


class DiceAndSoftBCE(nn.Module):
    def __init__(self, bce_weight, dice_weight, label_smoothing, reduction='mean'):
        super(DiceAndSoftBCE, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)
        self.dice = DiceLoss(sigmoid=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target_bce = target * positive_smoothed_labels + (1 - target) * negative_smoothed_labels

        loss = self.bce_weight * self.bce_with_logits(input, target_bce) + self.dice_weight * self.dice(input, target)
        return loss

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, bce_weight, dice_weight):
        super(BCEDiceLoss, self).__init__()
        logging.info(f'bce_weight: {bce_weight}, dice_weight: {dice_weight}')
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.dice = DiceLoss(sigmoid=True)

    def forward(self, input, target):
        return self.bce_weight * self.bce(input, target) + self.dice_weight * self.dice(input, target)

def get_loss(name, **kwargs):
    if name == 'dice':
        logging.info('Using DiceLoss')
        return DiceLoss(sigmoid=True)
    elif name == 'softdice':
        logging.info('Using SoftDiceLoss')
        return SoftDiceLoss(**kwargs)
    elif name == 'diceandsoftbce':
        logging.info('Using DiceAndSoftBCE')
        return DiceAndSoftBCE(**kwargs)
    elif name == 'bceloss':
        logging.info('Using BCELoss')
        return nn.BCELoss(**kwargs)
    elif name == 'bcewithlogitsloss':
        logging.info('Using BCEWithLogitsLoss')
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == 'bcediceloss':
        logging.info('Using BCEDiceLoss')
        return BCEDiceLoss(**kwargs)
    else:
        raise ValueError(f'Invalid loss name: {name}')
