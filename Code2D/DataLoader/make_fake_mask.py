from typing import Literal
import numpy as np
from PIL import Image
import random
import os

def generate_arc_mask(direction_angle: int = 0, 
                      size: int = 400, 
                      move: int = 200, 
                      radius: int = 300,
                      mode: Literal['large', 'small'] = 'large'
                      ):
    """
    生成一个指定大小的黑白图像，圆外为白色，圆内为黑色。

    参数:
    direction_angle (float): 圆的方向角度
    size (int): 图像的大小（宽度和高度）
    move (int): 圆心相对于图像中心的偏移量
    radius (int): 圆的半径
    mode (str): 图像类型，'large'表示大掩膜，'small'表示小掩膜

    返回:
    np.ndarray: 生成的黑白掩膜图像
    """
    # 创建一个空白的图像，初始颜色为黑色
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # 计算图像的中心
    center = (size // 2, size // 2)
    
    # 计算圆心
    arc_center = (center[0] + move * np.cos(np.radians(direction_angle)), 
                  center[1] + move * np.sin(np.radians(direction_angle)))
    
    # 生成掩膜图像
    y, x = np.ogrid[:size, :size]
    distance_from_center = np.sqrt((x - arc_center[0]) ** 2 + (y - arc_center[1]) ** 2)
    mask[distance_from_center <= radius] = 255
    if mode =='small':
        mask[distance_from_center <= radius * 0.7] = 0  # 圆内改为黑色
    # 圆内改为黑色
    
    return mask

def main():
    """
    主函数，调用generate_arc_mask生成掩膜，并保存图像至指定路径，
    同时统计黑白像素的比例。
    """
    direction_angle = 0  # 圆的方向角度
    size = 400           # 图像的大小
    move = 200           # 圆心相对于图像中心的偏移量
    radius = 300         # 圆的半径
    save_path = 'd:\\WangDao\\2DTrainCode\\Code2D\\DataLoader\\arc_mask.png'  # 保存路径
    
    # 生成掩膜图像
    mask = generate_arc_mask(direction_angle, size, move, radius)
    
    # 统计黑白像素比例
    black_pixels = np.sum(mask == 0)
    white_pixels = np.sum(mask == 255)
    total_pixels = size * size
    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    
    print(f"黑色像素比例: {black_ratio:.2f}")
    print(f"白色像素比例: {white_ratio:.2f}")
    
    # 保存图像
    if save_path:
        img = Image.fromarray(mask)
        img.save(save_path)
        print(f"图像已保存至 {save_path}")
        
if __name__ == "__main__":
    main()
