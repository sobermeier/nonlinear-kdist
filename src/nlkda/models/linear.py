# coding=utf-8
"""Implementation of MRkNNCoP tree bound model."""
import logging
from typing import Tuple

import numpy
from scipy.spatial import ConvexHull

from .base import BoundModel

EPSILON = 1.0e-16


def compute_bounds(
    log_k: numpy.ndarray,
    log_d: numpy.ndarray
) -> numpy.ndarray:
    """
    Compute the optimal bounds as given in MRkNNCoP tree for all points.

    Reference:
        Achtert, E.; Böhm, C.; Kröger, P.; Kunath, P.; Pryakhin, A. & Renz, M.
        Efficient Reverse K-nearest Neighbor Search in Arbitrary Metric Spaces
        Proceedings of the 2006 ACM SIGMOD International Conference on Management of Data, ACM, 2006 , 515-526

    :param log_k: numpy.ndarray, dtype: numpy.float32, shape: (k,)
        log(1), ..., log(k_max)
    :param log_d: numpy.ndarray, dtype: numpy.float32, shape: (n, k)
        log_d[i, k] = log(nndist(x_i, k))
    :return: numpy.ndarray, dtype: numpy.float32, shape: (n, 2, 2)
        output out, where
            out[:, 0, 0]: offset of lower bound
            out[:, 0, 1]: slope of lower bound (>= 0)
            out[:, 1, 0]: offset of upper bound
            out[:, 1, 1]: slope of upper bound (>= 0)
    """
    # assert len(log_k) == settings.K_MAX
    n, k = log_d.shape

    # Pre-allocate for faster computation
    result = numpy.empty(shape=(n, 2, 2), dtype=numpy.float32)

    # iterate over points (convex hull computation non-broadcastable)
    for i in range(n):
        # Compute convex hull
        this_log_d = log_d[i, :]
        points = numpy.stack([log_k, this_log_d], axis=-1)
        hull = ConvexHull(points)

        # Decompose into upper and lower convex hull
        first_index = 0
        last_index = k - 1
        hull_indices = list(hull.vertices)
        start = hull_indices.index(first_index)
        hull_indices = hull_indices[start:] + hull_indices[:start]
        break_point = hull_indices.index(last_index)
        upper_indices = hull_indices[:break_point + 1]
        lower_indices = hull_indices[break_point:] + [0]
        upper_indices = numpy.asarray(upper_indices)
        lower_indices = numpy.asarray(lower_indices)

        # Store optimal bounds
        result[i, 0, :] = get_optimal_parameters(x=log_k, y=this_log_d, indices=upper_indices)
        result[i, 1, :] = get_optimal_parameters(x=log_k, y=this_log_d, indices=lower_indices)

    return result


def get_optimal_parameters(
    x: numpy.ndarray,
    y: numpy.ndarray,
    indices: numpy.ndarray
) -> numpy.ndarray:
    """
    Compute the optimal parameters for fitting the line in log-log space.

    Reference:
        Achtert, E.; Böhm, C.; Kröger, P.; Kunath, P.; Pryakhin, A. & Renz, M.
        Efficient Reverse K-nearest Neighbor Search in Arbitrary Metric Spaces
        Proceedings of the 2006 ACM SIGMOD International Conference on Management of Data, ACM, 2006 , 515-526

    :param x: numpy.ndarray, dtype: numpy.float32, shape: (k,)
        The x-coordinates (i.e. log k)
    :param y: numpy.ndarray, dtype: numpy.float32, shape: (k,)
        The y-coordinates
    :param indices: numpy.ndarray, dtype: numpy.int32, shape (l,)
        The indices of the points contained in the (upper/lower) convex hull (l <= k).
    :return: numpy.ndarray, dtype: numpy.float32, shape: (2,)
        The optimal parameters, (offset, slope)
    """
    # assert len(indices) > 1
    optimal_parameters = None
    optimal_mse = numpy.inf

    for i in range(len(indices) - 1):
        # Get coordinates of first point
        first_index = indices[i]
        first_x = x[first_index]
        first_y = y[first_index]

        # Get coordinates of second point
        j = i + 1
        second_index = indices[j]
        second_x = x[second_index]
        second_y = y[second_index]

        # Compute regression line
        delta_x = first_x - second_x
        delta_y = first_y - second_y
        slope = delta_y / delta_x
        offset = first_y - slope * first_x

        # Compute mse
        mse = numpy.sum(numpy.square(slope * x + offset - y))

        # Update optimal parameters
        if mse < optimal_mse:
            optimal_mse = mse
            optimal_parameters = numpy.asarray([offset, slope], dtype=numpy.float32)

    # assert optimal_parameters is not None

    return optimal_parameters


def coefficients_to_k_distance_bounds(
    coefficients: numpy.ndarray,
    log_k: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Given the coefficients of the bounds in log-log space, computes the bounds in normal distance space.

    :param coefficients: shape: (n, 2, 2)
            coefficients[:, 0, 0]: offset of lower bound
            coefficients[:, 0, 1]: slope of lower bound (>= 0)
            coefficients[:, 1, 0]: offset of upper bound
            coefficients[:, 1, 1]: slope of upper bound (>= 0)
    :param log_k: shape: (k,)
        The prepared log(k) values.

    :return: shape: (n, 2, k)
        The lower and upper bound.
    """
    log_bounds = coefficients[:, :, 0, None] + log_k[None, None, :] * coefficients[:, :, 1, None]
    bounds = numpy.exp(log_bounds).astype(numpy.float32)
    lower, upper = bounds[:, 0, :], bounds[:, 1, :]
    return lower, upper


class MRkNNCoPTreeBounds(BoundModel):
    """Implementation of the bound model of MRkNNCoP tree."""

    def __init__(self):
        self.coefficients = None
        self.log_k = None

    def fit(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        sample_weights: numpy.ndarray = None,
    ) -> None:
        """
        Fit the parameters of the model.

        :type sample_weights: dummy to fit interface
        :param x: The coordinates.
        :param y: The k-distances.

        :return: None.
        """
        self.log_k = numpy.log(1 + numpy.arange(y.shape[1]))
        log_k_distances = numpy.log(y + EPSILON)

        # Coefficients: shape: (n, 2, 2)
        self.coefficients = compute_bounds(log_k=self.log_k, log_d=log_k_distances)

    def predict_bounds(
        self,
        x: numpy.ndarray,
        is_predicted: bool = False,
        agg_point: bool = False
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self.coefficients is None or self.log_k is None:
            raise AssertionError('The model is not fitted.')
        logging.warning('Ignore x; assuming static setting!')
        return coefficients_to_k_distance_bounds(coefficients=self.coefficients, log_k=self.log_k)

    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        # Mean prediction
        mean_coefficients = self.coefficients.mean(axis=1)
        return mean_coefficients[:, 0, None] + mean_coefficients[:, 1, None] * self.log_k[None, :]

    def set_min_max(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        is_predicted: bool = False,
        agg_point: bool = False
    ) -> None:
        logging.warning(f"There is no effect of set_min_max on {self.__class__.__name__}")
