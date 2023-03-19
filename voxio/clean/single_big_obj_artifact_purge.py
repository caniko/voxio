from itertools import count
from logging import getLogger

import imagesize
import numpy as np
from pydantic import DirectoryPath, FilePath, validate_arguments
from scipy import ndimage
from scipy.ndimage import find_objects

from voxio.read import chunk_read_stack_images
from voxio.utils.io import cv2_read_any_depth, write_indexed_images_to_directory
from voxio.utils.misc import number_of_planes_loadable_to_memory

logger = getLogger(__name__)


def _volume_from_slices(*slices: slice) -> int:
    volume = 1
    for comp_slice in slices:
        volume *= comp_slice.stop - comp_slice.start
    return volume


def _read_and_purge_small_artifacts(image_path: FilePath) -> np.ndarray[bool, bool]:
    labeled, num_features = ndimage.label(cv2_read_any_depth(image_path))
    size_to_label = {
        _volume_from_slices(*slices): label
        for label, slices in zip(range(1, num_features + 1), find_objects(labeled))
        if slices
    }
    return labeled == size_to_label[max(size_to_label)]


@validate_arguments
def clear_everything_but_largest_object(image_paths: tuple[FilePath, ...], output_directory: DirectoryPath) -> None:
    counter = count()
    for cleaned_image_stack in chunk_read_stack_images(
        image_paths,
        number_of_planes_loadable_to_memory(
            imagesize.get(image_paths[0]),
            memory_tolerance=0.33,
            byte_mul=2,
        ),
        _read_and_purge_small_artifacts,
    ):
        write_indexed_images_to_directory(cleaned_image_stack, counter, output_directory, one_bit_image=True)
