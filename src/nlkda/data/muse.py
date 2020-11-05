"""Loading utilities for FastText dataset."""
import gzip
import io
import logging
import pathlib

import numpy

from .base import Dataset

logger = logging.getLogger(name=__name__)


class FastTextDataset(Dataset):
    """Dataset with fast-text vectors."""

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
    sha512 = "788ffbd8ee7be4394c6dd5bde20478603aef364a6085e4cf0f7a0495f6221c8120b55c92c480bf82483b0602e0a2810522f56756864eae728f17a3f739673bf7"
    short_name = "EN"
    distance = "cosine"
    n_max: int = 200_000

    @classmethod
    def preprocess(cls, file_path: pathlib.Path) -> numpy.ndarray:  # noqa: D102
        # cf. https://github.com/facebookresearch/MUSE/blob/d83a5b6031a9f3fb00e3193c9b14729fbdd54ff7/demo.ipynb
        x = numpy.empty(shape=(cls.n_max, 300), dtype=numpy.float32)
        with gzip.open(file_path, mode="rt", encoding="utf-8", newline="\n", errors="ignore") as f:
            next(f)
            for i, line in enumerate(f):
                if i >= cls.n_max:
                    break
                _word, vector = line.rstrip().split(' ', 1)
                x[i] = numpy.fromstring(vector, sep=' ', dtype=numpy.float32)
        return x


def load_vec(
    emb_path: pathlib.Path,
    n_max: int = 200_000,
) -> numpy.ndarray:
    """Load vectors from FastText file."""

    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = numpy.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == n_max:
                break
    return numpy.vstack(vectors)
