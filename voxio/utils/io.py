import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import DirectoryPath, FilePath
from pydantic_numpy import NpNDArray
from pydantic_numpy.typing import NpNDArrayBool


def cv2_read_any_depth(image_path: FilePath) -> np.ndarray:
    """
    Read an image using OpenCV with any depth.

    Parameters
    ----------
    image_path : FilePath
        The path to the image file.

    Returns
    -------
    np.ndarray
        The image data in the form of a numpy array.
    """
    return cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)


def read_binarize_rgb(image_path: FilePath) -> np.ndarray[bool, bool]:
    """
    Reads and binarizes an RGB image.

    Parameters
    ----------
    image_path : FilePath
        The path to the image file.

    Returns
    -------
    np.ndarray[bool, bool]
        The binarized image data in the form of a boolean numpy array.
    """
    return np.any(cv2_read_any_depth(image_path), axis=-1)


def compressed_png_save(image: NpNDArray, out_path: Path) -> None:
    """
    Saves a numpy array as a compressed png file.

    Parameters
    ----------
    image : NpNDArray
        The image data in the form of a numpy array.
    out_path : Path
        The path to save the compressed png file.
    """
    assert out_path.suffix == ".png"
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def one_bit_save(image: NpNDArrayBool, out_path: Path) -> None:
    """
    Saves a binary image array as a 1-bit png file.

    Parameters
    ----------
    image : NpNDArrayBool
        The binary image data in the form of a numpy array.
    out_path : Path
        The path to save the 1-bit png file.
    """
    Image.fromarray(image).save(out_path, bits=1, optimize=True)


def write_indexed_images_to_directory(
    images: Iterable[NpNDArray],
    index_iterator: Iterable[int],
    output_directory: DirectoryPath,
    one_bit_image: bool = False,
) -> None:
    """
    Writes a series of indexed images to a specified directory.

    Parameters
    ----------
    images : Iterable[NpNDArray]
        The images to write, in the form of a numpy array iterable.
    index_iterator : Iterable[int]
        The indices for each image.
    output_directory : DirectoryPath
        The directory to write the images to.
    one_bit_image : bool, optional
        If True, saves the images as 1-bit png files. If False (default),
        saves the images as compressed png files.
    """
    with ThreadPoolExecutor() as executor:
        for image_plane in images:
            executor.submit(
                one_bit_save if one_bit_image else compressed_png_save,
                image_plane,
                output_directory / f"{next(index_iterator)}.png",
            )


compiled_default_finder = re.compile(r"^.*d+")


def find_and_sort_images(
    image_dir: DirectoryPath, index_regex: Optional[str] = None, image_format: Optional[str] = None
) -> Iterable[FilePath]:
    """
    Finds and sorts image files in a directory according to an index defined in the filename.

    Parameters
    ----------
    image_dir : DirectoryPath
        The directory to search for image files.
    index_regex : str, optional
        The regular expression to extract the index from the filenames. Defaults to r"^.*d+", if not defined.
    image_format : str, optional
        The file extension to search for. Defaults to None, which means all files will be considered.

    Returns
    -------
    Iterable[FilePath]
        A sorted iterable of image file paths.
    """

    image_file_pattern = "*"
    if image_format:
        image_file_pattern += image_format

    index_finder = re.compile(index_regex) if index_regex else compiled_default_finder

    return sorted(Path(image_dir).glob(image_file_pattern), key=lambda n: int(index_finder.findall(n.stem)[0]))
