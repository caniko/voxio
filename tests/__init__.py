from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).absolute().parent
RESOURCES_DIR = TEST_ROOT / "resources"
SIX_STACK_DIR = RESOURCES_DIR / "6-stack"

IMAGE_6_STACK = np.array(
    [
        [[1, 0, 1, 1, 0], [1, 0, 1, 1, 0], [1, 0, 1, 1, 0]],
        [[1, 0, 1, 1, 0], [1, 0, 1, 1, 0], [1, 0, 1, 1, 0]],
        [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]],
        [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]],
        [[1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1]],
        [[1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1]],
    ],
    dtype=bool,
)

# [1, 0, 1, 1, 0]
# [1, 0, 1, 0, 0]
# [1, 1, 1, 0, 1]
