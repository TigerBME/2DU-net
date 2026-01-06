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
                 w_proj_neg: float = 1.0,
                 w_pseudo: float = 1.0,
                 w_fg: float = 0.5,          # 新增：前景存在性约束权重
                 min_fg_ratio: float = 0.05, # 新增：最小前景比例
                 sigmoid: bool = True):
        super().__init__()
        self.low_th = low_th
        self.high_th = high_th
        self.w_proj_neg = w_proj_neg
        self.w_pseudo = w_pseudo
        self.w_fg = w_fg
        self.min_fg_ratio = min_fg_ratio
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

        # -------- A. 投影=0 → 强负样本 --------
        # MIP=0 → 切片必定无前景
        mask_proj_neg = (proj_label < 0.5).float()
        loss_proj_neg = (
            F.binary_cross_entropy(
                y_pred_sig, torch.zeros_like(y_pred_sig), reduction="none"
            ) * mask_proj_neg
        ).mean()

        # -------- B. 置信度伪标签 --------
        mask_high = (y_pred_sig > self.high_th).float()
        mask_low = (y_pred_sig < self.low_th).float()
        mask_conf = (mask_high + mask_low) * (1 - mask_proj_neg)

        pseudo_label = mask_high
        loss_pseudo = (
            F.binary_cross_entropy(
                y_pred_sig, pseudo_label, reduction="none"
            ) * mask_conf
        ).mean()

        # -------- C. 前景存在性约束（关键新增） --------
        mask_proj_pos = (proj_label > 0.5).float()
        if mask_proj_pos.sum() > 0:
            fg_ratio = (y_pred_sig * mask_proj_pos).sum() / (mask_proj_pos.sum() + 1e-6)
            loss_fg = F.relu(self.min_fg_ratio - fg_ratio) ** 2
        else:
            loss_fg = torch.tensor(0.0, device=y_pred.device)

        # -------- D. 融合损失 --------
        total_loss = (
            self.w_proj_neg * loss_proj_neg
            + self.w_pseudo * loss_pseudo
            + self.w_fg * loss_fg
        )

        return total_loss

