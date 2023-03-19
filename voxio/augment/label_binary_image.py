import json
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import Generator, Iterable

import compress_pickle
import numpy as np
from numpy import ScalarType
from pydantic import BaseModel, DirectoryPath, Field, FilePath, validate_arguments
from pydantic_numpy import NDArray
from scipy import ndimage
from yaspin import yaspin

from voxio.caching import CachingInfo
from voxio.read import chunk_read_stack_images
from voxio.utils.io import cv2_read_any_depth, write_indexed_images_to_directory
from voxio.utils.misc import get_image_dimensions, number_of_planes_loadable_to_memory

defaultdict_set = partial(defaultdict, set)


class StateOfLabel(BaseModel):
    chunk_size: int
    volume_index: int | None
    bad_object_locations: list[tuple[int, int, int]] = Field(default_factory=list)

    chunk_depths: list[int, ...] = Field(default_factory=list)
    object_id_to_locs: dict[int, set[int]] = Field(default_factory=defaultdict_set)

    volume_index_to_must_map_later: dict[int, set[tuple[int, int]]] = Field(
        default_factory=defaultdict_set, description="tuple[old, new]"
    )

    def add_loc_to_ids(self, ids: Iterable[int], loc: int) -> None:
        for oid in ids:
            self.object_id_to_locs[int(oid)].add(loc)

    def current_chunk_depth(self, volume_index: int) -> int:
        return sum(self.chunk_depths[:volume_index])

    def plane_image_id_range(self, volume_index: int) -> Iterable[int]:
        start = self.current_chunk_depth(volume_index)
        return iter(range(start, start + self.chunk_depths[volume_index]))


def remove_objects_that_contain_other_objects_from_labeled(labeled: NDArray) -> tuple[NDArray, NDArray]:
    """
    Caused by improper annotation. In VAST, most likely because of
    annotating at a high mip-level, which leads to gaps in the lower mips,
    which translates to extracted values.
    """
    bad_object_locations = []
    for object_slice in ndimage.find_objects(labeled):
        if not object_slice:
            continue

        unique_labels = np.unique(labeled[object_slice])
        if len(unique_labels) <= 2:
            continue

        bad_object_locations.append(np.array([(s.stop - s.start) / 2.0 for s in object_slice]))

        for label in unique_labels:
            if label:  # != 0
                labeled[labeled == label] = 0

    return ndimage.label(labeled)[0], np.array(bad_object_locations)


@validate_arguments
def label_binary_image(
    image_paths: tuple[FilePath, ...],
    output_directory: DirectoryPath,
    less_than_or_eq_254: bool = False,
):
    caching_info = CachingInfo(data_directory=output_directory)
    state = (
        compress_pickle.load(caching_info.state_path)
        if caching_info.state_path.exists()
        else StateOfLabel(
            chunk_size=number_of_planes_loadable_to_memory(
                get_image_dimensions(image_paths[0]),
                memory_tolerance=0.33,
                byte_mul=2 if less_than_or_eq_254 else 3,
            )
        )
    )
    return main_label_binary_image(
        image_generator=chunk_read_stack_images(
            image_paths,
            chunk_size=state.chunk_size,
            image_reader=cv2_read_any_depth if less_than_or_eq_254 else cv2_read_any_depth,
            offset=caching_info.number_of_arrays,
        ),
        output_directory=output_directory,
        state=state,
        np_data_type=np.uint8 if less_than_or_eq_254 else np.uint16,
    )


@yaspin(text="Labeling binary images...")
def main_label_binary_image(
    image_generator: Generator[np.ndarray, None, None],
    output_directory: DirectoryPath,
    state: StateOfLabel,
    clean_unbound: bool = True,
    np_data_type: ScalarType = np.uint16,
) -> None:
    def save_state() -> None:
        compress_pickle.dump(state, caching_info.state_path)

    output_directory = Path(output_directory)
    caching_info = CachingInfo(data_directory=output_directory)

    queue = deque([np.load(caching_info.array_file_paths[-1])["v"]]) if caching_info.array_file_paths else deque()

    for volume_index, volume_chunk in enumerate(image_generator):
        state.volume_index = volume_index
        upcoming = ndimage.label(volume_chunk)[0].astype(np_data_type)
        if clean_unbound:
            upcoming, bad_objects = remove_objects_that_contain_other_objects_from_labeled(upcoming)
            upcoming = upcoming.astype(np_data_type)
            if np.any(bad_objects):
                state.bad_object_locations.extend(np.add(bad_objects, [state.current_chunk_depth(volume_index), 0, 0]))

        queue.append(upcoming)
        state.chunk_depths.append(len(upcoming))

        if len(queue) < 2:
            continue

        first_argwhere = queue[1] != 0
        queue[1][first_argwhere] = queue[1][first_argwhere] + np.max(queue[0])
        del first_argwhere

        bottom_plane_1st = queue[0][-1]

        top_plane_2nd = queue[1][0]
        top_objects = ndimage.find_objects(top_plane_2nd)

        max_on_first = np.max(queue[1])

        previous_volume_index = state.volume_index - 1
        two_steps_back_index = state.volume_index - 2

        for object_slices in top_objects:
            if not object_slices:
                continue
            slice_of_previous_plane = bottom_plane_1st[object_slices]
            if not np.any(slice_of_previous_plane):
                continue

            overlapping_labels = deque(np.unique(slice_of_previous_plane))
            if overlapping_labels[0] == 0:
                overlapping_labels.popleft()

            if not overlapping_labels:
                continue

            unique_on_2nd_top_object_slice = np.unique(top_plane_2nd[object_slices])
            match len(unique_on_2nd_top_object_slice):
                case 1:
                    old_label_on_first = unique_on_2nd_top_object_slice[0]
                    assert old_label_on_first != 0
                case 2:
                    old_label_on_first = unique_on_2nd_top_object_slice[1]
                case _:
                    raise ValueError("Too many labels on the object slice")

            first = overlapping_labels.popleft()
            queue[1][queue[1] == old_label_on_first] = first
            queue[1][queue[1] == max_on_first] = old_label_on_first
            max_on_first -= 1

            for overlapping_label in overlapping_labels:
                if overlapping_label in queue[0][0] and previous_volume_index >= 1:
                    state.volume_index_to_must_map_later[two_steps_back_index].add((overlapping_label, first))
                queue[0][queue[0] == overlapping_label] = first

        state.add_loc_to_ids(np.unique(queue[0]), previous_volume_index)
        np.savez(str(caching_info.volume_array_by_index(previous_volume_index)), v=queue.popleft())
        save_state()

    state.add_loc_to_ids(np.unique(queue[0]), state.volume_index)
    del state.object_id_to_locs[0]

    np.savez(str(caching_info.volume_array_by_index(state.volume_index)), v=queue.pop())

    del volume_chunk, upcoming, queue, bottom_plane_1st, slice_of_previous_plane, top_plane_2nd

    """
    By scanning from the top we ensure the loading of volumes are minimal;
    scanning backwards, lets newer jobs trickle down. 
    """

    for volume_index, to_map_records in sorted(state.volume_index_to_must_map_later.items(), reverse=True):
        array_path = str(caching_info.volume_array_by_index(volume_index))
        array = np.load(array_path)["v"]

        for old, new in to_map_records:
            state.object_id_to_locs[old].remove(volume_index)
            if not state.object_id_to_locs[old]:
                del state.object_id_to_locs[old]

            if old not in array:
                continue

            if old in array[0] and volume_index > 0:
                state.volume_index_to_must_map_later[volume_index - 1].add((old, new))
            array[array == old] = new

            save_state()

        np.savez(array_path, v=array)

    save_state()

    loc_to_old_to_new_id = defaultdict(dict)
    for sequential_id, object_id in enumerate(state.object_id_to_locs, start=1):
        if sequential_id == object_id:
            continue
        for loc in state.object_id_to_locs[object_id]:
            loc_to_old_to_new_id[loc][object_id] = sequential_id

    for volume_index in range(state.volume_index + 1):
        array = np.load(str(caching_info.volume_array_by_index(volume_index)))["v"]

        for old, new in loc_to_old_to_new_id[volume_index].items():
            array[array == old] = new

        write_indexed_images_to_directory(
            array,
            state.plane_image_id_range(volume_index),
            output_directory,
        )

    with open(output_directory / "info.json", "w") as out_json:
        json.dump(
            {
                "bad_object_locations": state.bad_object_locations,
                "total_objects": len(state.object_id_to_locs),
            },
            out_json,
        )
