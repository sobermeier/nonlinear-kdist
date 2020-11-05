"""Test for models."""
import unittest
from typing import Any, Mapping, Optional, Type

import numpy
from sklearn.tree import DecisionTreeRegressor

from nlkda.models.base import BoundModel, KAsInputWrapper, KAsOutputWrapper
from nlkda.models.linear import MRkNNCoPTreeBounds
from nlkda.utils import all_knn

EPSILON = 1.0e-06


class BoundModelTests:
    """Generic tests for BoundModel."""

    #: The number of test data points
    n: int = 13
    #: The dimension of test data points
    d: int = 2
    #: The number of nearest neighbors
    k: int = 3

    #: The data points, shape: (n, d)
    x: numpy.ndarray
    #: The k-distances, shape: (n, k)
    y: numpy.ndarray

    #: The tested class
    cls: Type[BoundModel]
    #: keywords for instantiation
    kwargs: Optional[Mapping[str, Any]] = None

    #: The test instance
    instance: BoundModel

    def setUp(self) -> None:
        # generate some random data
        self.x = numpy.random.uniform(size=(self.n, self.d)).astype(dtype=numpy.float32)
        self.y = all_knn(x=self.x, k=self.k, batch_size=self.n, distance="minkowski")
        # instantiate
        self.instance = self.cls(**(self.kwargs or dict()))

    def _fit(self):
        self.instance.fit(x=self.x, y=self.y)

    def test_fit(self):
        """Test the fit method."""
        self._fit()

    def test_predict_bounds(self):
        """Test the predict bounds method without point aggregation."""
        self._fit()
        self.instance.set_min_max(x=self.x, y=self.y, is_predicted=False)
        lower, upper = self.instance.predict_bounds(x=self.x, is_predicted=False, agg_point=False)

        # verify shape
        assert lower.shape == self.y.shape
        assert upper.shape == self.y.shape

        # verify data type
        assert lower.dtype == numpy.float32
        assert upper.dtype == numpy.float32

        # verify value range
        assert (lower <= upper).all()
        assert ((lower - EPSILON) <= self.y).all()
        assert (self.y <= (upper + EPSILON)).all()


class MRkNNCoPTreeTests(BoundModelTests, unittest.TestCase):
    """Tests for MRkNNCoPTree."""

    cls = MRkNNCoPTreeBounds


class KAsInputWrapperTests(BoundModelTests, unittest.TestCase):
    """Tests for KAsInputWrapper."""

    cls = KAsInputWrapper
    kwargs = dict(
        base=DecisionTreeRegressor()
    )


class KAsOutputWrapperTests(BoundModelTests, unittest.TestCase):
    """Tests for KAsOutputWrapper."""

    cls = KAsOutputWrapper
    kwargs = dict(
        base=DecisionTreeRegressor()
    )
