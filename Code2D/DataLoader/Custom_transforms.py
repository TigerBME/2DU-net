import torch
import random
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import numpy as np
from .make_fake_mask import generate_arc_mask

# 自定义数据增强类
class RandomFlip:
    def __init__(self, p:float=0.5, direction:str='horizontal'):
        self.p = p
        self.direction = direction

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError("There should not be more than 2 input tensors.")
        if torch.rand(1).item() < self.p:
            if self.direction == 'horizontal':
                results = [TF.hflip(x) for x in args]
            else:
                results = [TF.vflip(x) for x in args]
            return results[0] if len(results) == 1 else tuple(results)
        return args[0] if len(args) == 1 else tuple(args)


class RandomRotate:
    def __init__(self, p:float=0.5, max_angle:int=30):
        self.p = p
        self.max_angle = max_angle

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError("There should not be more than 2 input tensors.")
        if torch.rand(1).item() < self.p:
            angle = torch.empty(1).uniform_(-self.max_angle, self.max_angle).item()
            results = [TF.rotate(x, angle) for x in args]
            return results[0] if len(results) == 1 else tuple(results)
        return args[0] if len(args) == 1 else tuple(args)


class AddGaussianNoise:
    def __init__(self, p:float=0.5, mean:float=0.0, var:float=0.05):
        self.p = p
        self.mean = mean
        self.std = var ** 0.5

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError("There should not be more than 2 input tensors.")
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(args[0]) * self.std + self.mean
            results = [x + noise for x in args]
            return results[0] if len(results) == 1 else tuple(results)
        
        return args[0] if len(args) == 1 else tuple(args)


class RandomEnhance:
    def __init__(self, p: float = 0.5, max_value: float = 2.0, enabel_color: bool = False):
        self.p = p
        self.max_value = max_value
        self.enabel_color = enabel_color

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError(f"{__class__.__name__}:There should not be more than 2 input tensors.")
        
        if len(args) > 1:
            raise ValueError(f"{__class__.__name__}:This is only for image, not for label.")
        
        if random.random() < self.p:
            value = random.uniform(-self.max_value, self.max_value)
            if self.enabel_color:
                # 使用颜色增强
                random_seed = random.randint(1, 4)
            else:
                # 灰度图，只使用亮度增强
                random_seed = random.randint(1, 3)
            
            results = []
            for img in args:
                if random_seed == 1:
                    enhancer = ImageEnhance.Brightness(img)
                elif random_seed == 2:
                    enhancer = ImageEnhance.Sharpness(img)
                elif random_seed == 3:
                    enhancer = ImageEnhance.Contrast(img)
                else:
                    enhancer = ImageEnhance.Color(img)

                results.append(enhancer.enhance(value))
            
            return results[0] 
        # 不操作，直接返回
        return args[0] 


class RandomMask:
    def __init__(self, 
                 p_large: float = 0.3, # 大型掩膜的概率
                 p_small: float = 0.2, # 小型掩膜的概率
                 angle_range: list = [0, 360], # 弧度范围
                 move_range: list = [170, 180], # 移动范围
                 radius_range: list = [320, 340], # 半径范围
                 ):
        """
        随机为输入图像添加弧形掩膜。
        """
        self.p_large = p_large
        self.p_small = p_small
        self.angle_range = tuple(angle_range)
        self.move_range = tuple(move_range)
        self.radius_range = tuple(radius_range)

    def __call__(self, *args):
        """
        对输入的图像（或图像与标签）添加弧形掩膜。
        支持 PIL.Image.Image 输入类型。
        """
        if len(args) == 0 or len(args) > 2:
            raise ValueError("Input should be one or two arrays.")

        # 获取第一个图像来确定尺寸和类型
        img = args[0]
        
        # 检查图像尺寸
        if hasattr(img, 'shape'):
            size = img.shape[-1] if len(img.shape) == 3 else img.shape[0]
        elif hasattr(img, 'size'):
            size = img.size[0]  # PIL Image 的尺寸
        else:
            raise ValueError("Unsupported image type for size detection.")
            
        assert size == 400, "Input size should be 400."

        # 随机决定是否应用掩膜
        r = random.random()
        if r >= (self.p_large + self.p_small):
            # 不应用掩膜,概率为1-p_large-p_small
            return args[0] if len(args) == 1 else tuple(args)
        
        elif r < self.p_large:
            # 应用大型掩膜，概率为p_large
            mode = 'large'
            angle = random.randint(*self.angle_range)
            move = random.randint(*self.move_range)
            radius = random.randint(*self.radius_range)

            mask = generate_arc_mask(direction_angle=angle, size=size, 
                                    move=move, radius=radius,
                                    mode=mode)
        else:
            # 应用小型掩膜，概率为p_small
            mode ='small'
            angle = random.randint(*self.angle_range)
            move = random.randint(*self.move_range)
            radius = random.randint(*self.radius_range)

            mask = generate_arc_mask(direction_angle=angle, size=size, 
                                    move=move, radius=radius,
                                    mode=mode)
        mask = mask.astype(np.float32) / 255.0  # 转为0~1的float掩膜

        results = []
        for x in args:
            # PIL Image处理
            x_np = np.array(x)  # 转换为numpy数组
            # 根据图像模式处理mask
            if x.mode in ['L', 'P']:  # 灰度图
                masked_np = x_np * mask
            elif x.mode in ['RGB', 'RGBA']:  # 彩色图
                # 为每个通道应用相同的mask
                mask_3d = np.stack([mask] * (3 if x.mode == 'RGB' else 4), axis=-1)
                masked_np = x_np * mask_3d
            else:
                raise ValueError(f"Unsupported PIL image mode: {x.mode}")
                
            # 转换回PIL Image，保持原数据类型
            if x_np.dtype == np.uint8:
                masked_np = masked_np.astype(np.uint8)
                results.append(Image.fromarray(masked_np))
                
            else:
                raise TypeError(f"Unsupported input type: {type(x)}. "
                              f"Supported types: numpy.ndarray, torch.Tensor, PIL.Image.Image")

        return results[0] if len(results) == 1 else tuple(results)  
    

class RandomGamma:
    def __init__(self, p: float = 0.5, gamma_range: tuple = (0.7, 1.5)):
        """
        随机 Gamma 校正（非线性亮度调整）
        :param p: 应用该增强的概率
        :param gamma_range: Gamma 值的采样范围，(min, max)，通常 (0.7, 1.5)
        """
        self.p = p
        self.gamma_min, self.gamma_max = gamma_range

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError(f"{__class__.__name__}:There should not be more than 2 input tensors.")

        if len(args) > 1:
            raise ValueError(f"{__class__.__name__}:This is only for image, not for label.")

        if torch.rand(1).item() < self.p:
            # 只操作图像，不操作标签
            gamma = torch.empty(1).uniform_(self.gamma_min, self.gamma_max).item()
            img = args[0]
            
            # 假设输入 img 是 [H, W] 或 [1, H, W] 的 Tensor，值域为 [0, 255]
            # 先归一化到 [0, 1]，做 Gamma，再恢复到 [0, 255]
            was_uint8 = (img.dtype == torch.uint8)
            if was_uint8:
                img = img.float()
            
            img_norm = img / 255.0
            img_gamma = torch.pow(img_norm.clamp(min=1e-8), gamma)  # 避免 0^gamma 导致 NaN
            img_out = img_gamma * 255.0
            img_out = torch.clamp(img_out, 0, 255)
            
            if was_uint8:
                img_out = img_out.to(torch.uint8)
            
            return img_out
        
        return args[0]
    

