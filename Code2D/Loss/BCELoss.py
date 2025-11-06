# Loss/bce_loss.py
import torch.nn as nn
import torch

class BCELoss(nn.Module):
    def __init__(self, weight:float=None, reduction:str="mean", pos_weight:float=None):
        """
        BCE 损失函数封装
        
        参数:
            weight (float): 类别权重
            reduction (str): 归约方式
            pos_weight (float): 正样本权重
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            weight=weight, 
            reduction=reduction,
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
    
    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor)->torch.Tensor:
        return self.bce(y_pred, y_true)