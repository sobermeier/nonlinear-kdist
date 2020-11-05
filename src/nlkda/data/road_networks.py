"""Road Network datasets."""
import pathlib

import numpy
import pandas

from .base import Dataset

URL = "https://www.cs.utah.edu/~lifeifei/research/tpq/"


class RoadNetworkDataset(Dataset):
    """Base class for road network datasets."""

    distance = "minkowski"

    @classmethod
    def preprocess(cls, file_path: pathlib.Path, seed: int = 42) -> numpy.ndarray:  # noqa: D102
        df = pandas.read_csv(file_path, sep=' ', header=None, names=['Node ID', 'x', 'y'])
        values = df.loc[:, ['x', 'y']].values.astype(numpy.float32)
        # deterministic shuffle
        numpy.random.seed(seed=seed)
        numpy.random.shuffle(values)
        return values


class OldenburgDataset(RoadNetworkDataset):
    """The Oldenburg road network dataset."""

    url = URL + "OL.cnode"
    short_name = "OL"
    sha512 = "23528eda1d43bf83eaaa4385306c671ba3d4c14f25ff5ad8877f35e2c6a0cb8d1ea514421e9e3e379757c4fbb2ba8748adaa3889ff97ea6213945b46984eafde"


class CaliforniaDataset(RoadNetworkDataset):
    """The California road network dataset."""

    url = URL + "cal.cnode"
    short_name = "CAL"
    sha512 = "64a818a0992b69ebb2e95ee5679a88000057b40c2a73b023ba3b4b3a9fe6de89a278654466a60c9ec1cac370af637f48744d74934e52131c70395120b5441ac9"


class NorthAmericaDataset(RoadNetworkDataset):
    """The North America road network dataset."""

    short_name = "NA"
    url = URL + "NA.cnode"
    sha512 = "c0c3a6a908fe1e8a08712f462edfde764896cb7a5a31bb814642020911dc27465a1f204b58ee6ed4ff51e7a7b4d399f5b8c7bd050040221cf296bda2f1926a21"
