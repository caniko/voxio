from collections import deque
from logging import getLogger

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
from voxio.utils.io import cv2_read_any_depth
from voxio.utils.misc import number_of_planes_loadable_to_memory, sort_indexed_dict_keys_to_value_list

logger = getLogger(__name__)


"""
We find the largest slice across all chunks, and use the slice reference for chunks that are above and below
with respect to z (the chunk normal vector).
"""


def _read_and_purge_small_artifacts(npy_path: FilePath, label_to_keep: int) -> np.ndarray[bool, bool]:
    array = np.load(str(npy_path))


@validate_arguments
def clear_everything_but_largest_object(image_paths: tuple[FilePath, ...], output_directory: DirectoryPath) -> None:
    def _read_and_size_to_slice(idx: int, image_path: FilePath):
        labeled, num_features = ndimage.label(cv2_read_any_depth(image_path))
        chunk = ChunkInfo(
            chunk_index=idx,
            labeled=labeled,
            label_to_slice={label: slices for label, slices in zip(range(1, num_features + 1), find_objects(labeled))},
        )
        max_volume_to_chunk_idx[chunk.max_volume] = chunk.max_volume
        chunk.dump(caching.cache_directory, str(idx), pickle=True)

    caching = CachingInfo(working_directory=output_directory)

    max_volume_to_chunk_idx = {}
    chunk_size = number_of_planes_loadable_to_memory(
        imagesize.get(image_paths[0]),
        memory_tolerance=0.5,
        byte_mul=2,
    )

    deque(chunk_read_stack_images(image_paths, chunk_size, _read_and_size_to_slice), maxlen=0)

    number_of_chunks

    max_volume_chunk_idx = max_volume_to_chunk_idx[max(max_volume_to_chunk_idx)]
    start_chunk = ChunkInfo.load(caching.cache_directory, max_volume_chunk_idx)
