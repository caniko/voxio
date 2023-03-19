from functools import lru_cache

import itk
import numpy as np
from numpy import ScalarType
from pydantic_numpy import NDArray

from voxio.utils.typings import TupleSlice


def compute_new_z_resolution(number_of_planes: int, spacing: int) -> int:
    return (
        # The image planes
        number_of_planes
        # Number of spacers
        + (number_of_planes - 1) * spacing
    )


@lru_cache
def generate_spacing_array(yx_shape: TupleSlice, spacing: int, data_type: ScalarType) -> NDArray:
    return np.zeros((spacing, *yx_shape), dtype=data_type)


def add_inter_spacing(stack: NDArray, spacing_array: NDArray) -> NDArray:
    spaced_stack = []
    for plane in stack:
        spaced_stack.append(np.expand_dims(plane, axis=0))
        spaced_stack.append(spacing_array)
    spaced_stack.pop()
    return np.concatenate(spaced_stack)


def morphological_interpolation_max_resolution_spacing(
    labeled_stack: NDArray, ceiled_inter_stack_voxel_distance: int
) -> NDArray:
    return interpolate_spaced_array(
        add_inter_spacing(
            labeled_stack,
            generate_spacing_array(
                labeled_stack.shape[1:],
                ceiled_inter_stack_voxel_distance,
                labeled_stack.dtype,
            ),
        )
    )


def interpolate_spaced_array(spaced_array: NDArray) -> NDArray:
    return itk.GetArrayFromImage(itk.morphological_contour_interpolator(itk.GetImageFromArray(spaced_array)))
