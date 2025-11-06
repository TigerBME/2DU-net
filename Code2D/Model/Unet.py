from typing import Dict
from monai.networks.nets import UNet
import torch

class ModelConfigError(Exception):
    """自定义模型配置异常"""
    pass

def _create_unet(config: Dict) -> UNet:
    """创建UNet的具体实现"""
    try:
        return UNet(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=config["channels"],
            strides=config["strides"],
            num_res_units=config.get("num_res_units", 0),     # 默认不启用残差
            dropout=config.get("dropout", 0.0),               # 默认无dropout
            act=config.get("act", "RELU"),                    # 默认激活函数
            norm=config.get("norm", "INSTANCE")               # 默认归一化
        )
    except Exception as e:
        raise ModelConfigError(f"Failed to create UNet: {str(e)}")
    

def main():
    """测试Unet模型"""
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
    # 创建模型
    model = _create_unet(DEFAULT_CONFIG_UNET)
    
    # 创建测试输入 (batch_size=1, channels=1, height=400, width=400)
    input_tensor = torch.randn(1, 1, 400, 400)
    print(f"\n输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    output = model(input_tensor)
    print(f"输出张量形状: {output.shape}")

    #用 summary 展示模型
    from torchsummary import summary
    summary(model, (1, 400, 400))



if __name__ == "__main__":
    main()
