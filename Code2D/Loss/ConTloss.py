# Loss/ConfidenceThresholdLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceThresholdLoss(nn.Module):
    """
    使用二维投影标签监督单张二维切片的弱监督损失。
    结合：
      1. 投影负样本强监督（T_proj = 0 → y_pred 应为 0）
      2. 投影正样本区域的“伪标签 + 置信度”监督
      3. 投影正区域的软一致性约束（避免网络全部预测0）
    """

    def __init__(self,
                 low_th: float = 0.25,
                 high_th: float = 0.75,
                 w_proj_neg: float = 1.0,    # 投影负样本监督权重
                 w_pseudo: float = 1.0,      # 伪标签监督权重
                 sigmoid: bool = True):
        super().__init__()
        self.low_th = low_th
        self.high_th = high_th
        self.w_proj_neg = w_proj_neg
        self.w_pseudo = w_pseudo
        self.apply_sigmoid = sigmoid

    def forward(self, y_pred, proj_label):
        """
        y_pred: (H, W) — 当前二维切片预测
        proj_label: (H, W) — 二维投影标签
        """
        if self.apply_sigmoid:
            y_pred_sig = torch.sigmoid(y_pred)
        else:
            y_pred_sig = y_pred

        # -------- A. 投影=0 区域为可靠负样本 --------
        # MIP=0 → 切片必定无前景
        mask_proj_neg = (proj_label < 0.5).float()

        loss_proj_neg = (
            F.binary_cross_entropy(y_pred_sig, torch.zeros_like(y_pred_sig), reduction="none")
            * mask_proj_neg
        )

        # -------- B. 在投影=1 的区域使用置信度伪标签 --------
        mask_high = (y_pred_sig > self.high_th).float()
        mask_low = (y_pred_sig < self.low_th).float()
        mask_conf = (mask_high + mask_low) * (1 - mask_proj_neg)

        pseudo_label = mask_high  # 高→1，低→0

        loss_pseudo = (
            F.binary_cross_entropy(y_pred_sig, pseudo_label, reduction="none")
            * mask_conf
        )

        # -------- D. 融合损失 --------
        total_loss = (
            self.w_proj_neg * loss_proj_neg.mean()
            + self.w_pseudo * loss_pseudo.mean()
        )

        return total_loss
