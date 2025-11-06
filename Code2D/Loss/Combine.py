# Loss/__init__.py
import importlib
import sys
from typing import Dict, List
import torch
from torch import nn
from .DiceLoss import DiceLoss
from .BCELoss import BCELoss
from .MSELoss import MSELoss

current_module = sys.modules[__name__]

class CombinedLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, y_pred, y_true):
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.losses):
            total_loss += weight * loss_fn(y_pred, y_true)
        return total_loss

def get_loss_function(loss_config: Dict) -> nn.Module:
    """
    配置格式说明：
    - 组合损失函数：
        {
            "name": "CombinedLoss",
            "loss": [
                {
                    "name": "DiceLoss",
                    "weight": 0.6,      # 该损失的权重
                    "args": {"smooth": 1e-5}  # 可选参数
                },
                {
                    "name": "BCELoss",
                    "weight": 0.4,
                    "args": {}
                }
            ]
        }
    """
    if loss_config["name"] == "CombinedLoss":
        # 验证组合损失配置结构
        if "loss" not in loss_config:
            raise ValueError("CombinedLoss requires 'loss' field in config")
        if not isinstance(loss_config["loss"], list):
            raise TypeError("'loss' field must be a list of loss configurations")
        
        losses = []
        weights = []
        
        for loss_cfg in loss_config["loss"]:
            # 校验每个损失配置的必须字段
            required_keys = {"name", "weight"}
            missing_keys = required_keys - set(loss_cfg.keys())
            if missing_keys:
                raise ValueError(f"Loss config missing required keys: {missing_keys}")
            
            # 动态获取损失类
            loss_class = getattr(current_module, loss_cfg["name"], None)
            if not loss_class:
                raise ValueError(f"Loss type '{loss_cfg['name']}' not found")
            
            # 实例化损失函数
            loss_instance = loss_class(**loss_cfg.get("args", {}))
            losses.append(loss_instance)
            weights.append(loss_cfg["weight"])
        
        return CombinedLoss(losses, weights)
    
    else:  
        raise ValueError("Unsupported loss function: {}".format(loss_config["name"]))
   