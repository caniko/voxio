from voxio.cli import find_and_sort_images
from voxio.workflows.label_binary_image import label_binary_image

label_binary_image(
    tuple(find_and_sort_images("/win_data/vast/outcast/spine/binary")),
    "/win_data/vast/outcast/spine/labeled",
)
