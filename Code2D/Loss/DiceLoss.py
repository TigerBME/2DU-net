# Loss/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth:float=1e-5, sigmoid:bool=True, reduction:str="mean"):
        """
        Dice 损失函数
        
        参数:
            smooth (float): 平滑系数，防止除以零
            sigmoid (bool): 是否对输出应用sigmoid
            reduction (str): 损失归约方式 ("mean" | "sum" | "none")
        """
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        # 展平预测和真实标签
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum() + self.smooth
        
        loss = 1 - (2. * intersection + self.smooth) / denominator
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss