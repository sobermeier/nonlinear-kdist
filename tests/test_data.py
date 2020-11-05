"""Tests for data loading."""
import pathlib
import unittest
from typing import Type

import numpy

from nlkda.data.base import Dataset
from nlkda.data.muse import FastTextDataset
from nlkda.data.road_networks import CaliforniaDataset, NorthAmericaDataset, OldenburgDataset
from nlkda.settings import K_MAX


class DatasetTests:
    """Common tests for datasets."""

    #: The tested class
    cls: Type[Dataset]

    #: The expected vector dimension.
    dim: int

    #: The expected distance function.
    exp_distance: str

    #: Whether this dataset does not use an index
    no_index: bool = False

    data_root = pathlib.Path("/mnt/data")

    def test_load(self):
        """Test dataset loading."""
        x = self.cls.load(data_root=self.data_root)

        # check type
        assert isinstance(x, numpy.ndarray)

        # check data type
        assert x.dtype == numpy.float32

        # check shape
        num_instances = x.shape[0]
        assert x.shape == (num_instances, self.dim)

    def test_distances(self):
        """Test distance loading."""
        x = self.cls.load(data_root=self.data_root)
        distances = self.cls.load_distances(data_root=self.data_root, x=x, batch_size=1024)

        # check type
        assert isinstance(distances, numpy.ndarray)

        # check shape
        n = x.shape[0]
        assert distances.shape == (n, K_MAX)

        # check data type
        assert distances.dtype == numpy.float32

        # check sorting
        numpy.testing.assert_equal(distances, numpy.sort(distances, axis=-1))

    def test_distance_function_name(self):
        """Test name of distance function."""
        assert self.cls.distance == self.exp_distance


class RoadNetworkTests(DatasetTests):
    """Common tests for road network datasets."""

    dim = 2
    exp_distance = "minkowski"


class OldenburgTests(RoadNetworkTests, unittest.TestCase):
    """Tests for Oldenburg dataset."""

    cls = OldenburgDataset


class CaliforniaTests(RoadNetworkTests, unittest.TestCase):
    """Tests for California dataset."""

    cls = CaliforniaDataset


class NorthAmericaTests(RoadNetworkTests, unittest.TestCase):
    """Test for North America dataset."""

    cls = NorthAmericaDataset


class FastTextTests(DatasetTests, unittest.TestCase):
    """Test for FastText dataset."""

    cls = FastTextDataset
    dim = 300
    exp_distance = "cosine"
    no_index = True
