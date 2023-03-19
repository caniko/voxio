import re
from math import floor
from typing import Any, Callable, Sequence

import imagesize
import numpy as np
import psutil
from pydantic import DirectoryPath, FilePath, validate_arguments


@validate_arguments
def get_image_paths(image_directory: DirectoryPath, image_format: str, sorting_key: Callable) -> tuple[FilePath, ...]:
    assert image_directory.is_dir()

    file_filter = f"*.{image_format}" if image_format else "*.*"
    return tuple(sorted(image_directory.glob(file_filter), key=sorting_key))


def get_number_indexed_image_paths(image_directory: DirectoryPath, image_format: str = "png") -> tuple[FilePath, ...]:
    finder = re.compile(r"\d+")
    return tuple(sorted(image_directory.glob(f"*.{image_format}"), key=lambda p: finder.findall(p.stem)[0]))


@validate_arguments
def number_of_planes_loadable_to_memory(
    plane_shape: Sequence[int], memory_tolerance: float = 1.0, byte_mul: int = 1
) -> int:
    return floor(psutil.virtual_memory().available * memory_tolerance / (np.multiply.reduce(plane_shape) * byte_mul))


def break_into_chunks(source: Sequence, chunk_size: int) -> list[Sequence, ...]:
    chunks = []
    start, stop = 0, chunk_size
    while stop < len(source):
        chunks.append(source[start:stop])
        start = stop
        stop += chunk_size
    chunks.append(source[start:])
    return chunks


def sort_indexed_dict_keys_to_value_list(key_index_dict: dict[float, Any]) -> list:
    return [v for _k, v in sorted(key_index_dict.items(), key=lambda kv: kv[0])]


def get_image_dimensions(image_path: FilePath) -> tuple[int, int]:
    return imagesize.get(image_path)
