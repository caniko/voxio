import numpy as np
from pydantic_numpy import NDArray


def crop_3d_image_to_content(
    image: NDArray, z_should_be_on_edges: bool = False
) -> tuple[NDArray, tuple[int, int, int]]:
    # Find the boundary of the object array, and crop the array accordingly
    zyx_defined_idxs = np.where(image)
    z_offset, y_offset, x_offset = np.min(zyx_defined_idxs, axis=1)
    z_bound, y_bound, x_bound = np.max(zyx_defined_idxs, axis=1)

    assert z_should_be_on_edges or (z_offset == 0 and z_bound == len(image) - 1), (
        f"Got z-  offset: {z_offset}; bound: {z_bound}. image length {len(image) - 1}. "
        f"The object must be defined throughout the entire z bound"
    )

    # Because of the aforementioned, I will use z in the bounding
    return (
        image[z_offset : z_bound + 1, y_offset : y_bound + 1, x_offset : x_bound + 1],
        (z_offset, y_offset, x_offset),
    )
