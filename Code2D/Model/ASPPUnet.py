"""
ASPP Unet Network.
"""
from __future__ import division
import torch
import torch.nn as nn
from typing import Dict

class ModelConfigError(Exception):
    """自定义模型配置异常"""
    pass


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPPModule, self).__init__()
        # 1x1卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        # 不同扩张率的空洞卷积
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        # 特征融合
        self.conv1x1_final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        
        # 全局平均池化并上采样
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        # 拼接所有特征
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print(f"out shape: {out.shape}")
        out = self.conv1x1_final(out)
        return out


class ASPPUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(ASPPUnet, self).__init__()
        self.enc_input = ResEncoder(out_channels, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.downsample = downsample()
        self.aspp = ASPPModule(512,512)
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)
        self.final = nn.Conv2d(32, in_channels, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # 应用ASPP模块
        # print(f"input_feature shape: {input_feature.shape}")
        aspp_feature = self.aspp(input_feature)
        # print(f"aspp_feature shape: {aspp_feature.shape}")

        # 解码部分
        up4 = self.deconv4(aspp_feature)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        return final


def _create_aspp_unet(config: Dict) -> nn.Module:
    """创建 ASPPUnet 的具体实现"""
    try:
        classes = config['in_channels']
        channels = config['out_channels']
        return ASPPUnet(classes, channels)
    except Exception as e:
        raise ModelConfigError(f"创建ASPPUnet模型失败: {str(e)}")


def main():
    """测试ASPPUnet模型"""
    # 创建模型
    model = ASPPUnet(in_channels=1, out_channels=1)
    
    # 打印模型结构
    # print("ASPPUnet模型结构:")
    # print(model)
    
    # 创建测试输入 (batch_size=1, channels=1, height=400, width=400)
    input_tensor = torch.randn(1, 1, 400, 400)
    print(f"\n输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    output = model(input_tensor)
    print(f"输出张量形状: {output.shape}")


if __name__ == "__main__":
    main()

