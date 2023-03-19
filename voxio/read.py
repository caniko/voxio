from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from math import ceil, floor
from pathlib import Path
from typing import Optional, Iterable, Callable, Sequence, Generator

import cv2
import numpy as np
from pydantic import FilePath
from skimage.io import imread as ski_imread

logger = getLogger(__name__)


def simple_read_stack_images(
    image_paths: Sequence[FilePath],
    image_reader: Callable[[Path | str], np.ndarray],
    parallel: bool = True,
) -> np.ndarray:
    return (
        parallel_read_stack_images(image_paths, image_reader)
        if parallel
        else np.array([image_reader(img_path) for img_path in image_paths])
    )


def chunk_read_stack_images(
    image_paths: Sequence[FilePath],
    image_reader: Callable[[Path | str], np.ndarray],
    chunk_size: int,
    parallel: bool = True,
) -> Generator[np.ndarray, None, None]:
    """

    :param image_paths:
    :param image_reader:
    :param chunk_size:
    :param parallel:
    :return:
    """
    number_of_mid_chunks_minus_1 = floor(len(image_paths) / chunk_size)
    chunks = [image_paths[:chunk_size]]
    for chunk_idx in range(1, number_of_mid_chunks_minus_1):
        preceding = chunk_idx * chunk_size
        proceeding = preceding + chunk_size
        chunks.append(image_paths[preceding:proceeding])
    else:  # after end
        chunks.append(image_paths[proceeding:])

    for image_paths_chunk in chunks:
        yield (
            parallel_read_stack_images(image_paths_chunk, image_reader)
            if parallel
            else np.array([image_reader(img_path) for img_path in image_paths_chunk])
        )


def parallel_read_stack_images(image_paths: Iterable[Path], image_reader: Callable):
    with ThreadPoolExecutor() as executor:
        return np.array([image for image in executor.map(image_reader, image_paths)])


def parallel_scan_stack_images(image_paths: Sequence[Path], image_reader: Callable, with_index: bool = False):
    map_args = [image_reader, image_paths]
    if with_index:
        map_args.append(range(len(image_paths)))

    with ThreadPoolExecutor() as executor:
        executor.map(*map_args)
