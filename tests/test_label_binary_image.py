
import numpy as np

from tests import SIX_STACK_DIR
from tests.utils import read_in_chunks_of_two
from voxio.workflows.label_binary_image import StateOfLabel, main_label_binary_image
from voxio.read import simple_find_read_images


def test_label_binary_image(tmp_path, labeled_6_stack):
    main_label_binary_image(
        read_in_chunks_of_two(SIX_STACK_DIR), tmp_path, StateOfLabel(chunk_size=2), np_data_type=np.uint8
    )
    labeled_stack = simple_find_read_images(tmp_path)
    assert np.all(labeled_6_stack == labeled_stack)
