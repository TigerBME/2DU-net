# Loss/mse_loss.py
import torch.nn as nn
import torch
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean", sigmoid: bool = False):
        """
        MSE 损失函数封装
        
        参数:
            reduction (str): 归约方式
            apply_sigmoid (bool): 是否在计算损失前应用 Sigmoid 函数
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.apply_sigmoid = sigmoid
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            y_pred = F.sigmoid(y_pred)
        return self.mse(y_pred, y_true)
