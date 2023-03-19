import logging
from collections import defaultdict
from functools import cached_property
from statistics import mean, stdev
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

from voxio.utils.misc import number_of_planes_loadable_to_memory
from voxio.utils.typings import TupleSlice
from voxio.volume_info import MAX_NUMBER_OF_UNITS_UINT8

logger = logging.getLogger(__file__)


class VolumeInfo(BaseModel):
    plane_dimension: tuple[int, int]
    unit_id_to_size_stack_sequence: list[dict[int, int], ...]
    unit_pair_to_min_distance: dict[frozenset[int], float] = Field(default_factory=dict)
    unit_id_to_yx_slices: dict[int, list[slice, slice]] = Field(default_factory=dict)

    # We use this default_factory when serializing from YAML, empty sets confuse Pydantic
    unit_ids_on_stack_edge: set[int] = Field(default_factory=set)

    minimum_z_depth: ClassVar[int] = 4

    class Config:
        keep_untouched = (cached_property,)

    def unit_ids_in_z_range(
        self,
        start: int,
        stop: int,
        force_uint8_compatibility: bool = False,
        chunked_unit_id: Optional[int] = None,
        unit_ids_to_skip: Optional[set[int]] = None,
        preferred_partial_unit_distance: float = 4.0,
    ) -> tuple[set[int], set[int], bool]:
        """
        Returns units that are either entirely in the z range, and units that are partially in it.
        The 3rd value returned states if the z-range is uint8 compatible
        """
        if self.unit_pair_to_min_distance is None:
            msg = "Unit pair distance must be computed to find unit IDs in range"
            raise AttributeError(msg)

        range_set = {i for i in range(start, stop + 1)}
        uint8_compatible = True
        whole, partially = set(), set()

        if chunked_unit_id:
            # This is a trick to force chunked_unit_id into the set in case we trigger the uint8 compatibility enforcer.
            # By doing this we ensure one additional removal of partially with respect to proximity to the closes whole
            whole.add(chunked_unit_id)

        for unit_id, (unit_start, unit_stop) in self.unit_id_to_z_range.items():
            if uint8_compatible and len(whole) + len(partially) == MAX_NUMBER_OF_UNITS_UINT8:
                uint8_compatible = False
            if (
                unit_ids_to_skip
                and unit_id in unit_ids_to_skip
                or unit_id in self.unit_ids_with_less_than_minimum_z_depth
            ):
                continue

            if unit_start in range_set and unit_stop in range_set:
                whole.add(unit_id)
            elif unit_start in range_set or unit_stop in range_set:
                partially.add(unit_id)

        number_of_whole = len(whole)

        if not uint8_compatible and number_of_whole <= MAX_NUMBER_OF_UNITS_UINT8:
            space_for_partials = MAX_NUMBER_OF_UNITS_UINT8 - number_of_whole

            partial_id_to_min_dist_whole = {}
            for partial_unit_id in partially:
                current_shortest_distance = float("inf")
                for pair, distance in self.unit_pair_to_min_distance.items():
                    if partial_unit_id in pair and current_shortest_distance > distance:
                        current_shortest_distance = distance

                partial_id_to_min_dist_whole[partial_unit_id] = current_shortest_distance

            partial_id_to_min_dist_whole = dict(
                sorted(
                    partial_id_to_min_dist_whole.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            )

            for partial_id, distance in partial_id_to_min_dist_whole.items():
                if len(partially) == space_for_partials:
                    logger.debug(
                        "Achieved uint8 compatibility by removing partial units that are not neighbors to whole units"
                    )
                    uint8_compatible = True
                    break
                if not force_uint8_compatibility and distance <= preferred_partial_unit_distance:
                    logger.debug("Failed to achieve uint8 compatibility, too large z-range, reverting to previous")
                    # We want to signal uint8 incompatibility if there are close partial units
                    # that would be removed for uint8 compatibility, and revert to a smaller version of the group
                    break

                partially.remove(partial_id)
            else:  # no break
                raise AssertionError("There should've been a break in this for loop, bug in uint8 compatibility loop")

        if chunked_unit_id:
            # Cleaning up after myself! See the above if chunked_unit_id for context.
            whole.remove(chunked_unit_id)

        return whole, partially, uint8_compatible

    @cached_property
    def number_of_image_planes_loadable_to_memory(self) -> int:
        return number_of_planes_loadable_to_memory(self.plane_dimension)

    @property
    def unit_id_stack_sequence(self) -> list[set[int], ...]:
        return [set(stack_dict.keys()) for stack_dict in self.unit_id_to_size_stack_sequence]

    @cached_property
    def unit_id_to_stack_locations(self) -> dict[int, list[int, ...]]:
        result = defaultdict(list)
        for plane_index, unit_id_to_size_on_plane in enumerate(self.unit_id_to_size_stack_sequence):
            for unit_id in unit_id_to_size_on_plane:
                result[unit_id].append(plane_index)
        return result

    @cached_property
    def unit_id_to_size(self) -> dict[int, int]:
        result = defaultdict(lambda: 0)
        for unit_id_to_size_on_plane in self.unit_id_to_size_stack_sequence:
            for unit_id, object_size_on_plane in unit_id_to_size_on_plane.items():
                result[unit_id] += int(object_size_on_plane)
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

    @cached_property
    def unit_id_to_gaps(self) -> dict[int, list[int, ...]]:
        result = {}
        for unit_id, locations in self.unit_id_to_stack_locations.items():
            if gaps := set(range(locations[0], locations[-1] + 1)).difference(locations):
                result[unit_id] = list(gaps)
        return result

    @cached_property
    def unit_id_to_z_range(self) -> dict[int, TupleSlice]:
        return dict(
            sorted(
                [
                    (unit_id, (locations[0], locations[-1]))
                    for unit_id, locations in self.unit_id_to_stack_locations.items()
                ],
                key=lambda kv: kv[1][1] - kv[1][0],
                reverse=True,
            )
        )

    @cached_property
    def unit_id_to_z_depth(self) -> dict[int, int]:
        result = {unit_id: start_stop[1] - start_stop[0] for unit_id, start_stop in self.unit_id_to_z_range.items()}

        # Sort by z-depth big -> small
        return dict(sorted(result.items(), key=lambda v: v[1], reverse=True))

    @cached_property
    def all_unit_ids(self) -> set[int]:
        return set(self.unit_id_to_z_depth)

    @cached_property
    def unit_ids_with_less_than_minimum_z_depth(self) -> set[int]:
        return set(unit_id for unit_id, z_depth in self.unit_id_to_z_depth.items() if z_depth < self.minimum_z_depth)

    @cached_property
    def unit_ids_to_isolate(self) -> set[int]:
        result = self.all_unit_ids.difference(self.unit_ids_with_less_than_minimum_z_depth)
        return result.difference(self.unit_pair_to_min_distance) if self.unit_pair_to_min_distance else result

    @cached_property
    def full_scan_grouping_maximum_z_distance_for_merge(self) -> int:
        return mean(self.unit_id_to_size.values()) - 2 * stdev(self.unit_id_to_size.values())
