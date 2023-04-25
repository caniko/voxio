from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from logging import getLogger
from statistics import mean
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
from voxio.utils.io import cv2_read_any_depth, write_indexed_images_to_directory_with_executor
from voxio.utils.misc import (
    number_of_planes_loadable_to_memory,
    sort_indexed_dict_keys_to_value_list,
    get_image_index_range,
    ndim_slice_contains_other,
    biggest_slice_from_two,
    ex_enumerate,
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
    mean_chunk_depth = mean((chunk.z_depth for chunk in chunk_sequence))

    max_volume_chunk_idx = max_volume_to_chunk_idx[max(max_volume_to_chunk_idx)]
    start_chunk = chunk_sequence[max_volume_chunk_idx]

    start_chunk.add_label_of_interest_top(start_chunk.largest_label)
    start_chunk.add_label_of_interest_bottom(start_chunk.largest_label)
    start_slice = start_chunk.label_to_slice[start_chunk.largest_label]

    """
    We need to find the object across the chunks. We go from the chunk with largest object to the top, A.
    After reaching the top we need to retrace our steps to make sure the connections to the edges
    that weren't included in the first pass (A) are added, B. We do the same by going from the bottom, and
    back to where we started in A.
    """

    # A: From the slice with the largest object in all the chunks to the top
    previous_top_slices = [start_slice[1:]]
    next_top_slices = []
    for chunk_idx, chunk in enumerate(chunk_sequence[max_volume_chunk_idx + 1 :], start=max_volume_chunk_idx + 1):
        for label, object_slice in chunk.label_to_start_slices.items():
            for slice_in_previous_top in previous_top_slices:
                if ndim_slice_contains_other(slice_in_previous_top, object_slice):
                    chunk.add_label_of_interest_bottom(label)
                    if label in chunk.label_to_end_slices:
                        next_top_slices.append(chunk.label_to_end_slices[label])
        previous_top_slices = copy(next_top_slices)
        next_top_slices = []

    # B: From the top to the bottom
    previous_bottom_slices = list(chunk.bottom_label_interest_to_object_slice.values())
    next_bottom_slices = []
    for chunk_idx, chunk in ex_enumerate(chunk_sequence[-1::-1], start=number_of_chunks - 1, step=-1):
        for label, object_slice in chunk.label_to_end_slices.items():
            for slice_in_previous_bottom in previous_bottom_slices:
                if ndim_slice_contains_other(slice_in_previous_bottom, object_slice):
                    chunk.add_label_of_interest_top(label)
                    if label in chunk.label_to_start_slices:
                        next_bottom_slices.append(chunk.label_to_start_slices[label])
        previous_bottom_slices = copy(next_bottom_slices)
        next_bottom_slices = []

    # C: From the bottom to where we started in A
    previous_top_slices = list(chunk.top_label_interest_to_object_slice.values())
    for chunk_idx, chunk in enumerate(chunk_sequence):
        for label, object_slice in chunk.label_to_start_slices.items():
            for slice_in_previous_top in previous_top_slices:
                if ndim_slice_contains_other(slice_in_previous_top, object_slice):
                    chunk.add_label_of_interest_bottom(label)

    # Unpack each chunk to compressed 1-bit images

    current_z_depth = 0
    with ThreadPoolExecutor(max_workers=round(mean_chunk_depth * 1.5)) as executor:
        for chunk in chunk_sequence:
            top_index = current_z_depth + chunk.z_depth
            write_indexed_images_to_directory_with_executor(
                executor=executor,
                images=chunk.labeled_without_background_labels,
                index_iterator=iter(range(current_z_depth, top_index)),
                one_bit_image=True,
                output_directory=output_directory
            )
            current_z_depth = top_index
