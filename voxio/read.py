from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path
from typing import Callable, Generator, Iterable, Sequence

import numpy as np
from pydantic import DirectoryPath, FilePath, validate_arguments

from voxio.utils.io import cv2_read_any_depth
from voxio.utils.misc import break_into_chunks, get_number_indexed_image_paths

logger = getLogger(__name__)


def read_stack_images(
    image_paths: Sequence[FilePath],
    image_reader: Callable[[Path | str], np.ndarray],
    parallel: bool = True,
) -> np.ndarray:
    return (
        parallel_read_stack_images(image_paths, image_reader)
        if parallel
        else np.array([image_reader(img_path) for img_path in image_paths])
    )


def simple_read_images(image_paths: Sequence[FilePath], parallel: bool = True) -> np.ndarray:
    return read_stack_images(image_paths, cv2_read_any_depth, parallel)


@validate_arguments
def simple_find_read_images(image_directory: DirectoryPath, *finder_args, parallel: bool = True) -> np.ndarray:
    return read_stack_images(
        get_number_indexed_image_paths(image_directory, *finder_args), cv2_read_any_depth, parallel
    )


def chunk_read_stack_images(
    image_paths: Sequence[FilePath],
    chunk_size: int,
    image_reader: Callable[[Path | str], np.ndarray],
    offset: int = 0,
    parallel: bool = True,
) -> Generator[np.ndarray, None, None]:
    for idx, image_paths_chunk in enumerate(break_into_chunks(image_paths, chunk_size)):
        if idx < offset:
            continue
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
