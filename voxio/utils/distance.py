from logging import getLogger
from math import sqrt
from time import sleep
from typing import Optional

import numpy as np
import psutil
from pydantic_numpy import NDArray
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries

logger = getLogger(__name__)


def image_feature_distance(
    labeled_array: NDArray, memory_headroom: Optional[float] = None
) -> dict[frozenset[int], float]:
    """
    Takes a labeled array as returned by scipy.ndimage.label and
    returns an intra-feature distance matrix.

    https://stackoverflow.com/a/37230147
    """
    borders = find_boundaries(labeled_array, mode="inner")
    unique_labels = np.unique(labeled_array)

    labels = labeled_array[borders]
    coords = np.argwhere(borders)
    del borders

    sorter = np.argsort(labels)
    labels = labels[sorter]
    coords = coords[sorter]

    if memory_headroom:
        while psutil.virtual_memory().available <= memory_headroom:
            sleep(2)

    sq_dists = cdist(coords, coords, "sqeuclidean")

    start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
    nonzero_vs_feat = np.minimum.reduceat(sq_dists, start_idx, axis=1)
    del sq_dists, sorter, coords

    feat_vs_feat = np.minimum.reduceat(nonzero_vs_feat, start_idx, axis=0)
    del nonzero_vs_feat

    i_upper, j_upper = np.triu_indices(feat_vs_feat.shape[0], k=1)

    return {
        frozenset((unique_labels[i], unique_labels[j])): sqrt(feat_vs_feat[i, j])
        # loop over the upper triangle indices and extract the pairwise distances
        for i, j in zip(i_upper, j_upper)
    }
