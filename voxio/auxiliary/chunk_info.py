from collections import defaultdict
from functools import cached_property

import numpy as np
import scipy.ndimage
from pydantic import DirectoryPath
from pydantic_numpy import NumpyModel, NDArrayBool, NDArray
from scipy import ndimage
from scipy.ndimage import find_objects


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
        self.max_volume: int = self.label_to_volume[self.largest_label]

        end_slices = ndimage.find_objects(labeled[-1])
        result = defaultdict(list)
        for object_slices in end_slices:
            result[map_sub_slice_to_label(object_slices, -1)].append(object_slices)

        self.label_to_end_slices: dict[int, list[tuple[slice, slice], ...]] = dict(result)

        end_slices = ndimage.find_objects(labeled[0])
        result = defaultdict(list)
        for object_slices in end_slices:
            result[map_sub_slice_to_label(object_slices, 0)].append(object_slices)

        self.label_to_start_slices: dict[int, list[tuple[slice, slice], ...]] = dict(result)

        self.cache_path = cache_directory / f"{chunk_index}.npz"
        np.savez(str(self.cache_path), labeled)

    @property
    def read_labeled(self) -> np.ndarray:
        return np.load(str(self.cache_path))
