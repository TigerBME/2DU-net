import torch
import random
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

class RandomRotate3D:
    def __init__(self, p:float=0.5, max_angle:int=15):
        self.p = p
        self.max_angle = max_angle

    def _get_rotation_matrix(self, angles:int):
        """生成三维旋转矩阵（欧拉角）"""
        cx, cy, cz = [torch.cos(angle) for angle in angles]
        sx, sy, sz = [torch.sin(angle) for angle in angles]

        return torch.tensor([
            [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
            [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
            [-sy,   cy*sx,            cy*cx]
        ])

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError("There should not be more than 2 input tensors.")
        if any(x.dim() != 4 for x in args):
            raise ValueError("Input tensors must be 4D (C, D, H, W)")
            
        if torch.rand(1).item() < self.p:
            angles = torch.deg2rad(torch.FloatTensor(3).uniform_(-self.max_angle, self.max_angle))
            matrix = self._get_rotation_matrix(angles)
            
            results = []
            for tensor in args:
                grid = torch.nn.functional.affine_grid(
                    matrix.unsqueeze(0).to(tensor.device),
                    tensor.unsqueeze(0).size(),
                    align_corners=False
                )
                results.append(torch.nn.functional.grid_sample(
                    tensor.unsqueeze(0),
                    grid,
                    align_corners=False
                ).squeeze(0))
            
            return results[0] if len(results) == 1 else tuple(results)
        return args[0] if len(args) == 1 else tuple(args)

class RandomFlip3D:
    def __init__(self, p:float=0.5, axes=['horizontal', 'vertical', 'depth']):
        self.p = p
        self.axes_mapping = {
            'horizontal': -1,  # 宽度轴
            'vertical': -2,    # 高度轴
            'depth': -3        # 深度轴（新增）
        }
        self.axes = [self.axes_mapping[ax] for ax in axes]

    def __call__(self, *args):
        if len(args) > 2:
            raise ValueError("There should not be more than 2 input tensors.")
        if any(x.dim() != 4 for x in args):
            raise ValueError("Input tensors must be 4D (C, D, H, W)")
            
        for axis in self.axes:
            if torch.rand(1).item() < self.p:
                args = [torch.flip(x, dims=[axis]) for x in args]
        return args[0] if len(args) == 1 else tuple(args)
   