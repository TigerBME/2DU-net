"""
Channel and Spatial AttentionUNet Network (CS-Net).
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv_block(x)
        out = out + residual  # 非inplace操作，避免计算图出错
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class AttentionGate(nn.Module):
    def __init__(self, gate_channel, connecting_channel, F_int):
        super(AttentionGate, self).__init__()
        # F_g: gating signal channels (from decoder deeper layer)
        # F_l: skip connection feature channels (from encoder)
        # F_int: intermediate channel size
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channel, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(connecting_channel, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gate):
        # x: skip connection feature map (from encoder) -> shape [B, C, H, W]
        # g: gating signal feature map (from decoder deeper) -> [B, Gate_C, H, W] 
        g1 = self.W_g(gate)         # channels=F_int
        x1 = self.W_x(x)            # channels=F_int
        # Ensure the gate and the input feature map have the same spatial size;
        psi = self.relu(g1 + x1)    # channels=F_int
        psi = self.psi(psi)         # channels=1
        # Broadcast psi to channel dimension
        return x * psi

class AttentionUnet(nn.Module):
    def __init__(self, classes, channels):
        super(AttentionUnet, self).__init__()
        # encoder as before
        self.enc_input = ResEncoder(channels, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.downsample = downsample()
        # decoder parts + deconv parts as before
        self.deconv4 = deconv(512, 256)
        self.decoder4 = Decoder(512, 256)
        self.deconv3 = deconv(256, 128)
        self.decoder3 = Decoder(256, 128)
        self.deconv2 = deconv(128, 64)
        self.decoder2 = Decoder(128, 64)
        self.deconv1 = deconv(64, 32)
        self.decoder1 = Decoder(64, 32)
        self.final = nn.Conv2d(32, classes, kernel_size=1)
        # Attention gates for each skip connection
        self.att3 = AttentionGate(gate_channel=256, connecting_channel=256, F_int=128)
        self.att2 = AttentionGate(gate_channel=128, connecting_channel=128, F_int=64)
        self.att1 = AttentionGate(gate_channel=64, connecting_channel=64, F_int=32)
        self.att0 = AttentionGate(gate_channel=32, connecting_channel=32, F_int=16)
        initialize_weights(self)

    def forward(self, x):
        enc0 = self.enc_input(x)       # channels=32
        down1 = self.downsample(enc0)

        enc1 = self.encoder1(down1)    # channels=64
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)    # channels=128
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)    # channels=256
        down4 = self.downsample(enc3)

        enc4 = self.encoder4(down4)    # channels=512

        # Decoder path with attention on skip connections
        up4 = self.deconv4(enc4)       # channels →256
        # attention on enc3 using gating signal up4
        att3 = self.att3(enc3, up4)
        merge4 = torch.cat((att3, up4), dim=1)  # channels 256+256 =512
        dec4 = self.decoder4(merge4)    # output channels=256

        up3 = self.deconv3(dec4)        # channels →128
        att2 = self.att2(enc2, up3)
        merge3 = torch.cat((att2, up3), dim=1) # 128+128=256
        dec3 = self.decoder3(merge3)    # output channels=128

        up2 = self.deconv2(dec3)        # →64
        att1 = self.att1(enc1, up2)
        merge2 = torch.cat((att1, up2), dim=1) #64+64=128
        dec2 = self.decoder2(merge2)    # output channels=64

        up1 = self.deconv1(dec2)        # →32
        att0 = self.att0(enc0, up1)
        merge1 = torch.cat((att0, up1), dim=1)  #32+32=64
        dec1 = self.decoder1(merge1)    # output channels=32

        final = self.final(dec1)
        return final

def _create_attention_unet(config: Dict) -> nn.Module:
    """创建 attention_unet(CSNet) 的具体实现"""
    try:
        return AttentionUnet(
            classes=config["out_channels"],
            channels=config["in_channels"]
        )
    except Exception as e:
        raise ModelConfigError(f"Failed to create AttentionUNet: {str(e)}")



if __name__ == '__main__':
    from torchsummary import summary
    # 创建模型并加载到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUnet(classes=1, channels=1).to(device)

    # 打印模型结构
    summary(model, (1, 400, 400))

    # ------------------- 测试显存占用 -------------------
    # 清空显存缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 构造一个batch=4的输入
    x = torch.randn(4, 1, 400, 400).to(device)

    # 前向传播
    with torch.no_grad():
        y = model(x)

    # 输出显存信息
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # 当前占用
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 2    # 已保留
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2   # 峰值占用

    print(f"\n当前显存占用：{allocated:.2f} MB")
    print(f"CUDA缓存保留：{reserved:.2f} MB")
    print(f"显存峰值占用：{peak:.2f} MB")
