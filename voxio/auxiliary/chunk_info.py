from collections import defaultdict
from functools import cached_property

import numpy as np
import scipy.ndimage
from pydantic import DirectoryPath
from pydantic_numpy import NumpyModel, NDArrayBool, NDArray
from scipy import ndimage
from scipy.ndimage import find_objects

from voxio.utils.misc import biggest_ndim_slice


def _volume_from_slices(*slices: slice) -> int:
    volume = 1
    for comp_slice in slices:
        volume *= comp_slice.stop - comp_slice.start
    return volume


class ChunkInfo:
    def __init__(self, image: np.ndarray, chunk_index: int, cache_directory: DirectoryPath):
        def map_sub_slice_to_label(sub_slice: tuple[slice, slice], z_index: int) -> int:
            labels = np.unique(labeled[z_index][sub_slice])
            if labels[0] == 0:
                assert len(labels) == 2
                return labels[1]
            assert len(labels) == 1
            return labels[0]

        labeled, num_features = ndimage.label(image)

        self.z_depth: int = len(labeled)
        self.chunk_index: int = chunk_index
        self.label_to_slice: dict[int, tuple[slice, slice, slice]] = {
            label: slices for label, slices in zip(range(1, num_features + 1), find_objects(labeled))
        }
        self.label_to_volume: dict[int, int] = {
            label: _volume_from_slices(*slices) for label, slices in self.label_to_slice.items()
        }

        largest_label, max_size = None, 0
        for label, size in self.label_to_volume.items():
            if size > max_size:
                max_size = size
                largest_label = label
        self.largest_label: int = largest_label
        self.max_obj_volume: int = self.label_to_volume[self.largest_label]
        self.max_obj_slice: tuple[slice, slice, slice] = self.label_to_slice[self.largest_label]

        self.max_start: tuple[tuple[slice]]

        end_slices = ndimage.find_objects(labeled[0])
        result = defaultdict(list)
        for object_slices in end_slices:
            result[map_sub_slice_to_label(object_slices, 0)].append(object_slices)

        self.label_to_start_slices: dict[int, list[tuple[slice, slice], ...]] = dict(result)

        end_slices = ndimage.find_objects(labeled[-1])
        result = defaultdict(list)
        for object_slices in end_slices:
            result[map_sub_slice_to_label(object_slices, -1)].append(object_slices)

        self.label_to_end_slices: dict[int, list[tuple[slice, slice], ...]] = dict(result)

        self.cache_path = cache_directory / f"{chunk_index}.npz"
        np.savez(str(self.cache_path), labeled)

        self.bottom_label_interest_to_object_slice: dict[int, tuple[slice, slice, slice]] = {}
        self.top_label_interest_to_object_slice: dict[int, tuple[slice, slice, slice]] = {}

    @property
    def read_labeled(self) -> np.ndarray:
        return np.load(str(self.cache_path))

    @property
    def labeled_without_background_labels(self):
        assert self.label_of_interest_to_object_slice

        labeled = self.read_labeled
        result = np.zeros_like(labeled, dtype=np.uint8)
        for label in self.label_of_interest_to_object_slice:
            result[labeled == label] = 1
        return result

    @property
    def max_zyx_slice_for_lois(self) -> list:
        if not self._label_interest_to_object_slice:
            return []

        all_slices = list(self._label_interest_to_object_slice.values())
        current_max = all_slices.pop()
        for cand_slices in all_slices:
            current_max = biggest_ndim_slice(current_max, cand_slices)

        return current_max

    @property
    def label_of_interest_to_object_slice(self) -> dict:
        return {**self._bottom_label_interest_to_object_slice, **self._top_label_interest_to_object_slice}

    def add_label_of_interest_bottom(self, label: int) -> None:
        self.bottom_label_interest_to_object_slice[label] = self.label_to_slice[label]

    def add_label_of_interest_top(self, label: int) -> None:
        self.top_label_interest_to_object_slice[label] = self.label_to_slice[label]
