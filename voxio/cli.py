from pathlib import Path
from typing import Optional

import click
from pydantic import DirectoryPath, validate_call

from voxio.utils.io import find_and_sort_images
from voxio.workflows.label_binary_image import label_binary_image

global_options = click.option("-i", "--input-dir", type=Path, help="The directory storing the images")(
    click.option("-o", "--output-dir", type=Path, help="The path to the directory to output the images")
)(click.option("-f", "--image-format", help="The format of the images"))(
    click.option(
        "-d",
        "--index-regex",
        help="Regex pattern to find the indices of the images in the stack",
        default=r"\d+",
    )
)


@click.group
def voxio_cli():
    pass


@voxio_cli.command
@global_options
@validate_call
def label_binary(image_dir: DirectoryPath, output_dir: Path, index_regex: str, image_format: Optional[str]) -> None:
    _output_dir_workflow(output_dir)
    label_binary_image(find_and_sort_images(image_dir))


def _output_dir_workflow(output_directory: DirectoryPath) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
