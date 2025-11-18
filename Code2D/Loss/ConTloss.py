# Loss/ConfidenceThresholdLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceThresholdLoss(nn.Module):
    def __init__(self,
                 low_th: float = 0.3,
                 high_th: float = 0.7,
                 reduction: str = "mean",
                 sigmoid: bool = True):
        """
        基于置信度阈值的伪标签损失函数
        y_true 会被接收但不使用。

        参数:
            low_th (float): 低置信度阈值 (< low_th → 伪标签=0)
            high_th (float): 高置信度阈值 (> high_th → 伪标签=1)
            reduction (str): 归约方式，仅支持 mean 或 sum
            apply_sigmoid (bool): 是否对 y_pred 先做 sigmoid
        """
        super().__init__()
        self.low_th = low_th
        self.high_th = high_th
        self.reduction = reduction
        self.apply_sigmoid = sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        参数:
            y_pred: 模型输出 (未 sigmoid 或 已 sigmoid)
            y_true: 输入的标签，但在本方法中不会使用
        """
        if self.apply_sigmoid:
            y_pred_sig = torch.sigmoid(y_pred)
        else:
            y_pred_sig = y_pred

        # -------- 1. 构造 mask：只使用高置信和低置信区域 --------
        mask_high = (y_pred_sig > self.high_th).float()
        mask_low = (y_pred_sig < self.low_th).float()
        mask = mask_high + mask_low  # 1: 参与训练, 0: 不参与

        # -------- 2. 构造伪标签 --------
        pseudo_label = (y_pred_sig > self.high_th).float()

        # -------- 3. 逐像素 BCE 损失 --------
        loss_pixel = F.binary_cross_entropy(
            y_pred_sig, pseudo_label, reduction="none"
        )

        # -------- 4. 只对 mask 中参与训练的像素计算 loss --------
        if mask.sum() == 0:
            # 没有任何像素满足阈值 → 返回 0 防止 NAN
            return torch.tensor(0.0, device=y_pred.device)

        loss = (loss_pixel * mask).sum() / (mask.sum() + 1e-6)

        return loss
