import torch
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms

def predict_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device, 
                  output_dir: str, threshold: float = 0.5) -> str:
    """执行模型预测并保存结果为BMP图像"""
    model.eval()
    
    # 定义张量转PIL图像的转换器
    tensor_to_pil = transforms.ToPILImage()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for inputs, output_paths in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu()
            
            # 二值化并转换为0-255的uint8格式
            preds = (preds > threshold).float() * 255  # [0 or 255]
            preds = preds.byte()  # 移除批次维度并转为uint8
            
            # 处理每个预测结果
            for pred, output_path in zip(preds, output_paths):
                # 转换张量为PIL图像
                pil_image = tensor_to_pil(pred.squeeze()) # 压缩通道维度
                
                # 保存为BMP
                pil_image.save(output_path)
                # print(f"Saved prediction: {output_path}")

    print(f"预测完成，结果保存在: {output_dir}")
    return output_dir