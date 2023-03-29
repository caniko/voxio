from functools import cached_property

from pydantic_numpy import NumpyModel, NDArrayBool, NDArray


def _volume_from_slices(*slices: slice) -> int:
    volume = 1
    for comp_slice in slices:
        volume *= comp_slice.stop - comp_slice.start
    return volume


class ChunkInfo(NumpyModel):
    chunk_index: int
    labeled: NDArray
    label_to_slice: dict[int, tuple[slice, slice, slice]]

    class Config:
        keep_untouched = (cached_property,)

    @cached_property
    def label_to_volume(self) -> dict[int, int]:
        return {label: _volume_from_slices(*slices) for label, slices in self.label_to_slice.items()}

    @cached_property
    def larges_label(self) -> int:
        largest_label, max_size = None, 0
        for label, size in self.label_to_volume.items():
            if size > max_size:
                max_size = size
                largest_label = label
        return largest_label

    @property
    def max_volume(self) -> int:
        return self.label_to_volume[self.larges_label]

    def top_
