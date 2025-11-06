import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer

# 自定义 PolyLR 类
class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    实现 Poly 学习率策略：lr = base_lr * (1 - epoch / max_epoch) ^ power
    """
    def __init__(self, optimizer: Optimizer, 
                 max_epoch: int, power: float = 0.9, last_epoch: int = -1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power
                for base_lr in self.base_lrs]
