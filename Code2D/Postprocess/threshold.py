import sys
import numpy as np
from skimage.filters import apply_hysteresis_threshold

class SingleThreshold:
    """
    Single-threshold binarization for segmentation post-processing.
    """

    def __init__(self, threshold: float = 0.5):
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold

    def __call__(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Args:
            prob_map (np.ndarray): Probability map in [0, 1].

        Returns:
            np.ndarray: Binary segmentation mask (0 or 1).
        """
        return (prob_map >= self.threshold).astype(np.uint8)

class HysteresisThreshold:
    """
    Hysteresis thresholding (dual-threshold binarization with connectivity constraint),
    implemented using scikit-image.
    """

    def __init__(self, low_threshold: float, high_threshold: float):
        assert 0.0 <= low_threshold < high_threshold <= 1.0
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Args:
            prob_map (np.ndarray): Probability map in [0, 1].

        Returns:
            np.ndarray: Binary segmentation mask (0 or 1).
        """
        binary_mask = apply_hysteresis_threshold(
            prob_map,
            low=self.low_threshold,
            high=self.high_threshold
        )
        return binary_mask.astype(np.uint8)

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


