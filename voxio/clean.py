from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path
from typing import Callable, Iterable, Optional
from pathlib import Path

import cv2
import numpy as np
from pydantic import DirectoryPath, FilePath, validate_arguments
from scipy import ndimage
from scipy.ndimage import find_objects, label
from skimage.io import imread as ski_imread

from voxio.read import simple_read_stack_images, chunk_read_stack_images

logger = getLogger(__name__)


def _volume_from_slices(*slices: Iterable[slice]) -> int:
    volume = 1
    for comp_slice in slices:
        volume *= comp_slice.stop - comp_slice.start
    return volume


def _read_and_clean_small_artifacts(image_volume: np.ndarray) -> np.ndarray[bool, bool]:
    labeled, num_features = ndimage.label(image_volume)

    object_slice_sequence = find_objects(labeled)
    size_to_label = {
        _volume_from_slices(*slices): label for label, slices in zip(range(1, num_features + 1), object_slice_sequence)
    }
    max_label = size_to_label[max(size_to_label)]

    return labeled[object_slice_sequence[max_label - 1]] == max_label


@validate_arguments
def clear_everything_but_largest_object(image_paths: Iterable[FilePath], output_directory: DirectoryPath) -> None:
    for idx, image_stack in enumerate(chunk_read_stack_images(image_paths, split_factor=5)):
        np.savez_compressed(comp_dir.parent / f"{comp_dir.stem}_{idx}.npz", stack=clean_smaller_artifacts(image_stack))
