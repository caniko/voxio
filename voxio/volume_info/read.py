import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from typing import Sequence

import compress_pickle
import numpy as np
import psutil
from imagesize import imagesize
from pydantic import FilePath
from pydantic_numpy import NDArray
from scipy.ndimage import find_objects

from voxio.read import cv2_read_any_depth, parallel_scan_stack_images
from voxio.utils.distance import image_feature_distance
from voxio.utils.misc import sort_indexed_dict_keys_to_value_list
from voxio.volume_info.volume_info import VolumeInfo

IndexAndImage = tuple[int, NDArray]

logger = logging.getLogger(__file__)


def _biggest_tuple_slice_from_two(slice_a: slice, slice_b: slice) -> slice:
    return slice(min(slice_a.start, slice_b.start), max(slice_a.stop, slice_b.stop))


def compute_stack_info(
    image_paths: Sequence[Path],
    stack_info_pickle_path: Path,
    record_objects_on_edge: bool = True,
    record_object_size: bool = True,
    record_object_slices: bool = True,
    record_minimum_pair_distance: bool = False,
    memory_headroom: float = 0.66,
) -> VolumeInfo:
    if stack_info_pickle_path.exists():
        return compress_pickle.load(stack_info_pickle_path)

    def plane_unique_image_label_to_size_computer(threaded_image_path: FilePath, idx: int) -> None:
        image = cv2_read_any_depth(threaded_image_path)
        assert plane_dimension == image.shape, f"Image stack missmatch: {plane_dimension} != {image.shape}"

        if record_objects_on_edge:
            ids_on_stack_edge.update(np.unique(image.T[0]))
            ids_on_stack_edge.update(np.unique(image.T[-1]))

        if record_object_size or record_object_slices:
            unique_labels = np.unique(image)
            if record_object_size:
                stack_idx_to_object_id_to_size_stack_sequence[idx] = {
                    object_id_in_image: int(np.sum(image == object_id_in_image)) for object_id_in_image in unique_labels
                }
            if record_object_slices:
                object_slices = find_objects(image)
                stack_idx_to_object_id_to_yx_slices[idx] = {
                    slice_object_id: [s for s in object_slices[slice_object_id - 1]]
                    for slice_object_id in unique_labels
                }

    plane_dimension = imagesize.get(image_paths[0])

    if record_minimum_pair_distance:

        def minimum_object_pair_distance_computer(threaded_image_path: FilePath) -> None:
            for pair, distance in image_feature_distance(cv2_read_any_depth(threaded_image_path)).items():
                if (
                    pair in object_pair_to_min_distance
                    and object_pair_to_min_distance[pair] > distance
                    or pair not in object_pair_to_min_distance
                ):
                    # Do not spawn more threads before memory usage is back down
                    object_pair_to_min_distance[pair] = distance

        component_pair_min_distance_cache = stack_info_pickle_path.parent / "min_pair_dist.pickle"

        object_pair_to_min_distance: dict[frozenset[int], float]
        if component_pair_min_distance_cache.exists():
            object_pair_to_min_distance = compress_pickle.load(component_pair_min_distance_cache)
        else:
            logger.debug(
                f"Performing pre-scan of 3D data, with memory headroom {round(memory_headroom) / 10**6} out of "
                f"available {round(psutil.virtual_memory().available) / 10**6} Mb"
            )
            headroom_overflow, object_pair_to_min_distance = False, {}
            with ThreadPoolExecutor() as executor:
                for image_path in image_paths:
                    while psutil.virtual_memory().available <= memory_headroom:
                        sleep(2)
                        headroom_overflow = True

                    executor.submit(minimum_object_pair_distance_computer, image_path)

                    if headroom_overflow:
                        # Wait an arbitrary amount of time before going into a bunch of extra batches
                        # The task that was submitted might be a big one also, we don't want to add more
                        # tasks till the memory has been emptied further
                        sleep(15)
                        headroom_overflow = False

    ids_on_stack_edge: set[int] = set()
    stack_idx_to_object_id_to_size_stack_sequence: dict[int, dict[int, int]] = {}
    stack_idx_to_object_id_to_yx_slices: dict[int, dict[int, tuple[slice, slice]]] = {}

    parallel_scan_stack_images(image_paths, plane_unique_image_label_to_size_computer, with_index=True)

    # We want to remove 0 from this set because it doesn't represent an object, but the void
    ids_on_stack_edge = {int(object_id_on_edge) for object_id_on_edge in ids_on_stack_edge}
    ids_on_stack_edge.remove(0)

    # We need to find the bounding box that fits the entire object
    # The z slice will be added in StackPreIsolationState.object_id_to_zyx_slices
    object_id_to_yx_slices = {}
    for plane_object_id_to_yx_slices in sort_indexed_dict_keys_to_value_list(stack_idx_to_object_id_to_yx_slices):
        for object_id, yx_slices in plane_object_id_to_yx_slices.items():
            if object_id not in object_id_to_yx_slices:
                object_id_to_yx_slices[object_id] = yx_slices
            else:
                new_y, new_x = yx_slices
                old_y, old_x = object_id_to_yx_slices[object_id]
                object_id_to_yx_slices[object_id] = (
                    _biggest_tuple_slice_from_two(old_y, new_y),
                    _biggest_tuple_slice_from_two(old_x, new_x),
                )

    return VolumeInfo(
        plane_dimension=plane_dimension,
        object_id_to_size_stack_sequence=sort_indexed_dict_keys_to_value_list(
            stack_idx_to_object_id_to_size_stack_sequence
        ),
        object_id_to_yx_slices=object_id_to_yx_slices,
        object_pair_to_min_distance=object_pair_to_min_distance,
        ids_on_stack_edge=ids_on_stack_edge,
    )
