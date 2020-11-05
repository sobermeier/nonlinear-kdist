"""General data loading classes."""
import hashlib
import logging
import pathlib
import urllib.request
from enum import Enum
from typing import Optional

import numpy
import tqdm

from ..settings import K_MAX
from ..utils import all_knn

logger = logging.getLogger(name=__name__)


class Dataset:
    """Base class for datasets."""

    #: The URL under which the dataset is available.
    url: str

    #: The sha-512 checksum.
    sha512: str

    #: A short name.
    short_name: str

    #: The name of the distance function to use with this dataset.
    distance: str

    @classmethod
    def load(
        cls,
        data_root: pathlib.Path,
        force: bool = False,
    ) -> numpy.ndarray:
        """
        Load the dataset, downloading the data files if necessary.

        :param data_root:
            The data root under which the files will be placed.
        :param force:
            Whether to enforce downloading the data.

        :return: shape: (n, d)
            The loaded dataset.
        """
        # dataset specific directories.
        data_root = cls._resolve_data_root(data_root=data_root)

        # download raw files if necessary
        raw_file_path = data_root / cls.url.rsplit("/")[-1]
        maybe_download(file_path=raw_file_path, url=cls.url, sha512=cls.sha512, force=force)

        # load from cache if possible
        preprocessed_file = data_root / "preprocessed.npy"
        if preprocessed_file.is_file():
            logger.info(f"Loading from cached file {preprocessed_file}")
            return numpy.load(str(preprocessed_file))

        # preprocess
        x = cls.preprocess(file_path=raw_file_path)
        # save
        numpy.save(str(preprocessed_file), x)
        return x

    @classmethod
    def _resolve_data_root(cls, data_root: pathlib.Path) -> pathlib.Path:
        data_root = data_root / cls.__name__
        data_root.mkdir(exist_ok=True, parents=True)
        return data_root

    @classmethod
    def preprocess(cls, file_path: pathlib.Path) -> numpy.ndarray:
        """
        Load the data from an existing file.

        :param file_path:
            The path of the downloaded file.

        :return: shape: (n, d)
            The dataset.
        """
        raise NotImplementedError

    @classmethod
    def load_distances(
        cls,
        data_root: pathlib.Path,
        x: numpy.ndarray,
        batch_size: int,
    ) -> numpy.ndarray:
        """
        Load distances or compute them.

        :param data_root:
            The data root.
        :param x: shape: (n, d)
            The data.
        :param batch_size:
            The batch size to use for brute-force distance calculation.
        :return:
            A tuple of (distances, indices) each of shape (n, k).
        """
        # resolve paths
        data_root = cls._resolve_data_root(data_root=data_root)
        distance_path = data_root / "distances.npy"

        # load distances
        if distance_path.is_file():
            distances = numpy.load(str(distance_path))
            return distances

        # compute distances
        distances = all_knn(x=x, k=K_MAX, batch_size=batch_size, distance=cls.distance)

        # save distances
        numpy.save(str(distance_path), distances)

        return distances


def maybe_download(
    file_path: pathlib.Path,
    url: str,
    sha512: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Download file if necessary.

    The download is necessary, if at least one of the following conditions is true
        * force is enabled
        * the file does not exist
        * the file exists and sha512 is provided, but the checksum does not match

    :param file_path:
        The file path under which to store the file.
    :param url:
        The URL from which to download the file.
    :param sha512:
        The checksum.
    :param force:
        Whether to force downloading the data.
    """
    do_download = True
    if file_path.is_file() and not force:
        logger.info(f"{file_path} already exists.")
        do_download = False
        if sha512 is not None:
            with file_path.open("rb") as f:
                sha512_ = hashlib.sha512(f.read()).hexdigest()
            if sha512_ != sha512:
                logger.info("Checksum did not match. Re-downloading.")
                do_download = True
    do_download = do_download or force
    if not do_download:
        return

    logger.info(f"Downloading from {url} to {file_path}.")

    # Show progress bar
    with tqdm.tqdm(unit='byte', unit_scale=True, desc='Download') as progress:
        def hook(count: int, size: int, total_size: int) -> None:  # pylint: disable=unused-argument
            if progress.n != total_size:
                progress.n = total_size
                progress.refresh()
            progress.update(size)

        urllib.request.urlretrieve(  # pylint: disable=unused-variable
            url=url,
            filename=file_path,
            reporthook=hook
        )


class DatasetEnum(str, Enum):
    """A enum for datasets."""
    OL = "OL"
    NA = "NA"
    CAL = "cal"
    WE_EN = "EN"


def get_dataset_size(dataset: DatasetEnum) -> int:
    """Get the number of data points."""
    if dataset == DatasetEnum.OL:
        return 6_105
    elif dataset == DatasetEnum.CAL:
        return 21_049
    elif dataset == DatasetEnum.NA:
        return 175_814
    elif dataset == DatasetEnum.WE_EN:
        return 200_000
    raise ValueError(dataset)

