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
