import sys
import numpy as np
from skimage.filters import apply_hysteresis_threshold
import torch
class SingleThreshold:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, prob_map: torch.Tensor) -> torch.Tensor:
        return (prob_map >= self.threshold).to(torch.uint8) * 255
class HysteresisThreshold:
    def __init__(self, low_threshold: float, high_threshold: float):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, prob_map: torch.Tensor) -> torch.Tensor:
        # 转为 NumPy
        prob_np = prob_map.cpu().numpy()
        mask_np = apply_hysteresis_threshold(prob_np, self.low_threshold, self.high_threshold)
        # 转回 Tensor 并转 0/255
        return torch.from_numpy(mask_np.astype(np.uint8)) * 255

class Remain_probability_process:
    def __call__(self, prob_map: torch.Tensor) -> torch.Tensor:
        '''
        ->[0-255]
        '''
        weight = prob_map.max()-prob_map.min()
        prob_map = (prob_map - prob_map.min()) / weight * 255.0
        return prob_map.to(torch.uint8)

def get_postprocess_function(config: dict):
    """
    根据配置字典获取后处理函数实例。

    配置格式示例：
    - 单阈值二值化：
        {
            "name": "SingleThreshold",
            "threshold": 0.5
        }
    - 双阈值滞后二值化：
        {
            "name": "HysteresisThreshold",
            "low_threshold": 0.3,
            "high_threshold": 0.7
        }   
    """
    name = config.pop("name")
    try:
        # 尝试获取类
        PostprocessClass = getattr(sys.modules[__name__], name)
        
        # 尝试创建实例
        try:
            postprocess = PostprocessClass(**config)
        except Exception as e:
            raise ValueError(f"Failed to instantiate {name} with config {config}: {str(e)}")
            
    except AttributeError as e:
        raise ValueError(f"Postprocess class '{name}' not found in module '{__name__}'. \
                         Available classes: {[cls for cls in dir(sys.modules[__name__]) if not cls.startswith('_')]}")

    return postprocess


