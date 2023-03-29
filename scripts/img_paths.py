from pathlib import Path

from voxio.augment.label_binary_image import label_binary_image
from voxio.cli import find_and_sort_images

output_path = Path("/win_data/vast/outcast/spine/labeled")
# output_path.mkdir(exist_ok=True)

label_binary_image(
    tuple(find_and_sort_images("/win_data/vast/outcast/spine/binary")),
    output_path,
)
