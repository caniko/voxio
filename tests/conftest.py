import shutil

import pytest
from scipy import ndimage

from tests import IMAGE_6_STACK, SIX_STACK_DIR
from voxio.utils.io import one_bit_save


@pytest.fixture
def labeled_6_stack():
    SIX_STACK_DIR.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(IMAGE_6_STACK):
        one_bit_save(img, SIX_STACK_DIR / f"{idx}.png")

    labeled, number_of_objects = ndimage.label(IMAGE_6_STACK)
    assert number_of_objects == 2

    yield labeled

    shutil.rmtree(SIX_STACK_DIR)
