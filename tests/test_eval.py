import unittest

import numpy
from scipy import spatial

from nlkda.eval import get_candidate_set_size


class CandidateSetSizeTests:
    """Base class for unittests for candidate set size computation."""
    num_samples: int = 13
    dim: int = 3
    k_max: int = 5
    batch_size: int = 7

    # The arrays
    x: numpy.ndarray
    dist: numpy.ndarray
    kdist: numpy.ndarray
    lower: numpy.ndarray
    upper: numpy.ndarray

    def setUp(self) -> None:
        """Allocate data; precompute distances."""
        self.x = numpy.random.uniform(size=(self.num_samples, self.dim))
        self.dist = spatial.distance.cdist(self.x, self.x, metric='euclidean')
        self.kdist = numpy.sort(self.dist)[:, :self.k_max]
        self.lower = self.kdist - numpy.random.uniform(low=0, high=0.2 * self.kdist.max(), size=self.kdist.shape)
        self.upper = self.kdist + numpy.random.uniform(low=0, high=0.2 * self.kdist.max(), size=self.kdist.shape)
        assert (self.lower <= self.kdist).all()
        assert (self.upper >= self.kdist).all()
        assert (self.upper >= self.lower).all()

    def _verify_candidates(
        self,
        num_candidates: numpy.ndarray,
    ):
        """Verify the number of candidates."""
        # check value range
        assert (num_candidates >= 0).all()
        assert (num_candidates < self.num_samples).all()

        # Comparison to brute-force
        exp_num_candidates = ((self.lower[:, :, None] <= self.dist[:, None, :]) & (self.dist[:, None, :] <= self.upper[:, :, None])).sum(axis=-1)
        numpy.testing.assert_array_equal(x=num_candidates, y=exp_num_candidates)


class RealBruteForceCandidateSetSizeTorchTests(CandidateSetSizeTests, unittest.TestCase):
    """unittest for real brute-force pytorch variant."""

    def test_get_candidate_set_size_bf_torch(self):
        num_candidates = get_candidate_set_size(
            lower=self.lower,
            upper=self.upper,
            x=self.x,
            distance="minkowski",
            batch_size=self.batch_size,
        )
        self._verify_candidates(num_candidates=num_candidates)
