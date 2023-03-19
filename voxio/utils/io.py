from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from pydantic import DirectoryPath, FilePath
from pydantic_numpy import NDArray, NDArrayBool


def cv2_read_any_depth(image_path: FilePath) -> np.ndarray:
    return cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)


def read_binarize_rgb(image_path: FilePath) -> np.ndarray[bool, bool]:
    return np.any(cv2_read_any_depth(image_path), axis=-1)


def compressed_png_save(image: NDArray, out_path: Path) -> None:
    # https://stackoverflow.com/a/60552336
    assert out_path.suffix == ".png"
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def one_bit_save(image: NDArrayBool, out_path: Path) -> None:
    Image.fromarray(image).save(out_path, bits=1, optimize=True)


def write_indexed_images_to_directory(
    images: Iterable[NDArray],
    index_iterator: Iterable[int],
    output_directory: DirectoryPath,
    one_bit_image: bool = False,
) -> None:
    with ThreadPoolExecutor() as executor:
        for image_plane in images:
            executor.submit(
                one_bit_save if one_bit_image else compressed_png_save,
                image_plane,
                output_directory / f"{next(index_iterator)}.png",
            )
