from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import find_objects


def _cv2_read_any_depth(image_path: Path) -> np.ndarray:
    return cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)


def read_binarize_rgb(image_path: Path) -> np.ndarray:
    return np.any(cv2.imread(str(image_path)), axis=2)
