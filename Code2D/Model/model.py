# model.py
from typing import Dict
import torch

from .Unet import _create_unet
from .AttentionUnet import _create_attention_unet
from .ASPPUnet import _create_aspp_unet

class ModelConfigError(Exception):
    """自定义模型配置异常"""
    pass

def create_model(config: Dict) -> torch.nn.Module:
    """
    根据配置字典创建 MONAI 支持的模型（UNet 或 CSNet）

    参数:
        config (Dict): 包含模型参数的字典

    返回:
        torch.nn.Module: 初始化好的模型实例
    """
    # 参数校验
    _validate_config(config)

    model_type = config["model_type"].lower()
    if model_type == "unet":
        return _create_unet(config)
    elif model_type == "csnet":
        return _create_attention_unet(config)
    elif model_type == "asppunet":
        return _create_aspp_unet(config)
    else:
        raise ModelConfigError(f"Unsupported model type: {config['model_type']}")

def _validate_config(config: Dict):
    """验证配置有效性"""
    required_keys_common = {
        "model_type", "spatial_dims", 
        "in_channels", "out_channels"
    }

    # UNet 特有参数
    required_keys_unet = {"channels", "strides"}

    missing_keys = required_keys_common - set(config.keys())
    if missing_keys:
        raise ModelConfigError(f"Missing required config keys: {missing_keys}")

    if config["spatial_dims"] not in (2, 3):
        raise ModelConfigError(f"Invalid spatial_dims: {config['spatial_dims']} (must be 2 or 3)")

    if config["model_type"].lower() == "unet":
        if not required_keys_unet.issubset(config.keys()):
            raise ModelConfigError("UNet config must contain 'channels' and 'strides'")
        if len(config["strides"]) != len(config["channels"]) - 1:
            raise ModelConfigError(
                f"Length mismatch: len(strides)={len(config['strides'])}, "
                f"len(channels)-1={len(config['channels'])-1}"
            )

# UNet 默认配置
DEFAULT_CONFIG_UNET = {
        "model_type": "unet",
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [64, 128, 256, 512],
        "strides": [1, 2, 2],
        "num_res_units": 2,
        "dropout": 0.1,
        "act": "RELU",
        "norm": "INSTANCE"
}

# AttentionUNet 默认配置
DEFAULT_CONFIG_ATTENTION_UNET = {
    "model_type": "csnet",
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1
}

# ASPPUnet 默认配置
DEFAULT_CONFIG_ASPP_UNET = {
    "model_type": "asppunet",
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1
}
