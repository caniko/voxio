from math import floor
from typing import Iterable, Callable

import numpy as np
import psutil
from pydantic import DirectoryPath, FilePath, validate_arguments


@validate_arguments
def get_image_paths(image_directory: DirectoryPath, image_format: str, sorting_key: Callable) -> Iterable[FilePath]:
    assert image_directory.is_dir()

    file_filter = f"*.{image_format}" if image_format else "*.*"
    return sorted(image_directory.glob(file_filter), key=sorting_key)


@validate_arguments
def number_of_planes_loadable_to_memory(plane_shape: Iterable[int], memory_tolerance: float = 1.0) -> int:
    return floor(psutil.virtual_memory().available * memory_tolerance / np.multiply.reduce(plane_shape))
