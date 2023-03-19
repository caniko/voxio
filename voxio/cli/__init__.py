import re
from pathlib import Path
from typing import Iterable, Optional

import click
from pydantic import DirectoryPath, FilePath, validate_arguments

from voxio.augment.label_binary_image import label_binary_image
from voxio.cli.settings import settings

global_options = click.option("-i", "--input-dir", type=Path, help="The directory storing the images")(
    click.option("-o", "--output-dir", type=Path, help="The path to the directory to output the images")
)(click.option("-f", "--image-format", help="The format of the images"))(
    click.option(
        "-d",
        "--index-regex",
        help="Regex pattern to find the indices of the images in the stack",
        default=settings.index_regex,
    )
)


def find_and_sort_images(
    image_dir: DirectoryPath, index_regex: str = settings.index_regex, image_format: Optional[str] = None
) -> Iterable[FilePath]:
    image_file_pattern = "*"
    if image_format:
        image_file_pattern += image_format

    index_finder = re.compile(index_regex)

    return sorted(Path(image_dir).glob(image_file_pattern), key=lambda n: int(index_finder.findall(n.stem)[0]))


def _output_dir_workflow(output_directory: DirectoryPath) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)


@click.group
def voxio_cli():
    pass


@voxio_cli.command
@global_options
@validate_arguments
def label_binary(image_dir: DirectoryPath, output_dir: Path, index_regex: str, image_format: Optional[str]) -> None:
    _output_dir_workflow(output_dir)
    label_binary_image(
        find_and_sort_images(
            image_dir,
        )
    )
