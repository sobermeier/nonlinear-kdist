"""Data loading."""
import pathlib
from typing import Tuple

import numpy

from .base import DatasetEnum
from .muse import FastTextDataset
from .road_networks import CaliforniaDataset, NorthAmericaDataset, OldenburgDataset


def get_data(
    dataset_enum: DatasetEnum,
    data_root: pathlib.Path,
    force_download: bool = False,
    batch_size: int = 1024,
) -> Tuple[numpy.ndarray, numpy.ndarray, str]:
    """
    Get dataset.

    :param dataset_enum:
        The selected dataset.
    :param data_root:
        The data root under which to store data files.
    :param force_download:
        Whether to enforce a re-download, even when files exist.
    :param batch_size:
        The batch size to use for computing the k-distances, if not present.

    :return:
        A tuple (x, y), where x.shape = (n, d). and y.shape = (n, k).
    """
    if dataset_enum == DatasetEnum.OL:
        cls = OldenburgDataset
    elif dataset_enum == DatasetEnum.CAL:
        cls = CaliforniaDataset
    elif dataset_enum == DatasetEnum.NA:
        cls = NorthAmericaDataset
    elif dataset_enum == DatasetEnum.WE_EN:
        cls = FastTextDataset
    else:
        raise ValueError(dataset_enum)
    x = cls.load(data_root=data_root, force=force_download)
    y = cls.load_distances(data_root=data_root, x=x, batch_size=batch_size)
    return x, y, cls.distance
