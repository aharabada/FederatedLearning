import torch

class DiceBCELoss(torch.nn.Module):
    """
    Calculate the Dice Loss and BCE Loss and return the sum of both.
    """
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        
    def forward(self, pred, target):
        # Ensure target is float and normalized
        target = target.float()
        if target.max() > 1 or target.min() < 0:
            target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 0.5 * bce_loss + 0.5 * dice_loss
    
    
def calculate_iou(pred, target):
    """
    Calculate the Intersection over Union (IoU) of the predicted mask and the target mask.
    """
    target = target.float()
    if target.max() > 1 or target.min() < 0:
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + 1e-5) / (union + 1e-5)