from collections import deque, defaultdict
from logging import getLogger
from typing import Sequence

import compress_pickle
import imagesize
import numpy as np
from numba import njit
from pydantic import DirectoryPath, FilePath, validate_arguments
from pydantic_numpy import NumpyModel, NDArrayBool
from scipy import ndimage
from scipy.ndimage import find_objects

from voxio.auxiliary.caching import CachingInfo
from voxio.auxiliary.chunk_info import ChunkInfo
from voxio.read import chunk_read_stack_images
from voxio.utils.io import cv2_read_any_depth, write_indexed_images_to_directory
from voxio.utils.misc import (
    number_of_planes_loadable_to_memory,
    sort_indexed_dict_keys_to_value_list,
    get_image_index_range, ndim_slice_contains_other, biggest_slice_from_two,
)

logger = getLogger(__name__)


"""
We find the largest slice across all chunks, and use the slice reference for chunks that are above and below
with respect to z (the chunk normal vector).
"""


@validate_arguments
def clear_everything_but_largest_object(image_paths: tuple[FilePath, ...], output_directory: DirectoryPath) -> None:
    def _read_and_size_to_slice(idx: int, image_path: FilePath):
        current_chunk = ChunkInfo(cv2_read_any_depth(image_path), idx, caching.cache_directory)
        max_volume_to_chunk_idx[current_chunk.max_volume] = idx
        chunk_idx_to_chunk[idx] = current_chunk

    caching = CachingInfo(working_directory=output_directory)

    chunk_idx_to_chunk = {}
    max_volume_to_chunk_idx = {}
    chunk_size = number_of_planes_loadable_to_memory(
        imagesize.get(image_paths[0]),
        memory_tolerance=0.45,
        byte_mul=2,
    )
    deque(chunk_read_stack_images(image_paths, chunk_size, _read_and_size_to_slice), maxlen=0)

    chunk_sequence = sort_indexed_dict_keys_to_value_list(chunk_idx_to_chunk)
    number_of_chunks = len(chunk_sequence)

    max_volume_chunk_idx = max_volume_to_chunk_idx[max(max_volume_to_chunk_idx)]
    start_chunk = chunk_sequence[max_volume_chunk_idx]

    start_slice = start_chunk.label_to_slice[start_chunk.largest_label]

    # From start to end
    previous_chunk = start_chunk
    previous_top = start_slice[1:]
    y_max, x_max = previous_top
    for chunk_idx, chunk in enumerate(chunk_sequence[max_volume_chunk_idx + 1:], start=max_volume_chunk_idx + 1):
        for label, object_slice in chunk.label_to_start_slices.items():
            if ndim_slice_contains_other(previous_top, object_slice):
                chunk.add_label_of_interest(label)
        y_max, x_max = chunk.max_zyx_slice_for_lois[1:]

    write_indexed_images_to_directory(start_chunk.read_labeled == start_chunk.largest_label, get_image_index_range())
