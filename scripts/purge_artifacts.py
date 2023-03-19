from pathlib import Path

from voxio.clean.single_big_obj_artifact_purge import (
    clear_everything_but_largest_object,
)
from voxio.cli import find_and_sort_images

for name in ("dendrite", "axon", "soma"):
    label_path = Path(f"/win_data/vast/max/{name}/labeled")
    label_path.mkdir(exist_ok=True)

    clear_everything_but_largest_object(tuple(find_and_sort_images(f"/win_data/vast/max/{name}/_binary")), label_path)
