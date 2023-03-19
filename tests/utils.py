import numpy as np
from pydantic import DirectoryPath

from voxio.utils.io import cv2_read_any_depth


def read_in_chunks_of_two(image_directory: DirectoryPath):
    for n, m in zip(range(0, 5, 2), range(1, 6, 2)):
        yield np.array(
            [
                cv2_read_any_depth(image_directory / f"{n}.png"),
                cv2_read_any_depth(image_directory / f"{m}.png"),
            ]
        )
