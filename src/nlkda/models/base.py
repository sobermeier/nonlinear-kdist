# coding=utf-8
"""
Models.
"""
from abc import ABC
from enum import Enum
from typing import Optional, Tuple, Union

import numpy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..settings import K_MAX


class FormulationWrapperEnum(Enum):
    K_AS_INPUT = "k_as_input"
    K_AS_OUTPUT = "k_as_output"


def is_multi_output(enum: FormulationWrapperEnum):
    return enum == FormulationWrapperEnum.K_AS_OUTPUT


def _k_as_input(
    params
):
    return KAsInputWrapper(**params)


def _k_as_output(
    params
):
    return KAsOutputWrapper(**params)


# map the inputs to the function blocks
wrapper_options = {
    FormulationWrapperEnum.K_AS_INPUT: _k_as_input,
    FormulationWrapperEnum.K_AS_OUTPUT: _k_as_output,
}


class BasicModel(ABC):
    """
    Basic model interface.
    """

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Predict k-distances.

        :param x: numpy.ndarray, shape: (n, d), dtype: numpy.float64
            The data points.

        :return: numpy.ndarray, shape: (n, ), dtype: numpy.float64
            The k-distances
        """
        raise NotImplementedError()

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> np.ndarray:
        """
        Predict k-distances.

        :param x: numpy.ndarray, shape: (n, d), dtype: numpy.float64
            The data points.

        :param y: numpy.ndarray, shape: (n, K_MAX), dtype: numpy.float64
            The target points.

        :param sample_weights: numpy.ndarray, shape: (n, K_MAX), dtype: numpy.float64
            Optional parameter sample_weights.

        :return: self, object
        """
        raise NotImplementedError()


class BoundModel(BasicModel, ABC):
    """
    Extension predicting bounds.
    """

    def set_min_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> None:
        """
        set min and max .

        :type agg_point: bool
            if true, bounds min/max is set to min/max aggregated over all points
            if false, bounds min/max is set to min/max aggregated over all k
        :param is_predicted: bool
            if true, x are already the predictions
            if false, x are data points and need to be predicted first

        :param x: numpy.ndarray, shape: (n, d), dtype: numpy.float64
            The data points.

        :param y: numpy.ndarray, shape: (n, K_MAX), dtype: numpy.float64
            The target points.

        :return: None
        """
        raise NotImplementedError()

    def predict_bounds(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict bounds of the k-distance.

        :param agg_point: bool
            if true, bounds are calculated by aggregating over points
            if false, bounds are calculated by aggregating over k
        :param x: numpy.ndarray, shape: (n, d), dtype: float
            The data points.

        :param is_predicted: bool
            if true, x are already the predictions
            if false, x are data points and need to be predicted first

        :return: [lb, ub] where
            lb: numpy.ndarray, shape: (n, K_MAX), dtype: float
                A lower bound.
            ub: numpy.ndarray, shape: (n, K_MAX), dtype: float
                An upper bound.
        """
        raise NotImplementedError()


# Formulation Wrappers for easy access.
class KAsInputWrapper(BoundModel):
    """
    KAsInputWrapper (single prediction): Overrides predict and fit methods of base model.
    Computes k-distances where k IS provided as input.
    """

    def __init__(
        self,
        base: BasicModel,
    ):
        self.base = base
        self.min_diff = 0
        self.max_diff = 0
        self.k = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> np.ndarray:
        """
        fits the base model with k as input in param x.

        :param sample_weights:
        :param x: numpy.ndarray, shape: (n, features), dtype: numpy.float64
            The data points. K as one input feature.

        :param y: numpy.ndarray, shape: (n, K_MAX), dtype: numpy.float64
            The target points.

        :return: base model object
        """
        self.k = y.shape[1]
        x_train, y_train = format_data(coord=x, skd=y, k=self.k)
        if sample_weights is not None:
            sample_weights = format_sample_weights(sample_weights)
        print(f'K INPUT WRAPPER: x shape: {x_train.shape}')
        print(f'K INPUT WRAPPER: y shape: {y_train.shape}')
        return self.base.fit(x_train, y_train, sample_weights)

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        predicts the k-distances.

        :param x: numpy.ndarray, shape: (n, features), dtype: numpy.float64
            The data points. K as one input feature.

        :return: numpy.ndarray, shape: (n, ), dtype: numpy.float64
            The k_distances directly predicted by the base model.
        """
        x_train = format_data(coord=x, k=self.k)
        y_pred = self.base.predict(x_train).astype(numpy.float32)
        return y_pred.reshape(-1).clip(min=0)

    def set_min_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ):
        y_train = y.reshape(-1)
        if is_predicted:
            y_pred = x
        else:
            y_pred = self.predict(x)

        # Point-wise prediction error
        diff = y_train - y_pred

        # Reshape to common form
        diff = diff.reshape(y.shape)

        if agg_point:
            # Compute minimal and maximal training error by aggregating over points
            self.min_diff = diff.min(axis=0)
            self.max_diff = diff.max(axis=0)
        else:
            # Compute minimal and maximal training error by aggregating over k
            self.min_diff = diff.min(axis=1)
            self.max_diff = diff.max(axis=1)

    def predict_bounds(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if is_predicted:
            y_pred = x
        else:
            y_pred = self.predict(x)
        pred_rs = np.reshape(y_pred, (-1, K_MAX))
        if agg_point:
            lower = pred_rs + self.min_diff[None, :]
            upper = pred_rs + self.max_diff[None, :]
        else:
            lower = pred_rs + self.min_diff[:, None]
            upper = pred_rs + self.max_diff[:, None]
        return lower.clip(min=0), upper.clip(min=0)


class KAsOutputWrapper(BoundModel, ABC):
    """
    KAsOutputWrapper (joint prediction): Overrides predict and fit methods of base model.
    Computes k-distances where k is NOT provided as input.
    """

    def __init__(
        self,
        base: BasicModel,
    ):
        self.base = base
        self.min_diff = 0
        self.max_diff = 0

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> np.ndarray:
        """
        fits the base model with k as output in param y.

        :param sample_weights:
        :param x: numpy.ndarray, shape: (n, features), dtype: numpy.float64
            The data points. Do not contain k.

        :param y: numpy.ndarray, shape: (n, K_MAX), dtype: numpy.float64
            The target points. K_MAX output values for each input entry.

        :return: base model object
        """
        print(f'K OUTPUT WRAPPER: x shape: {x.shape}')
        print(f'K OUTPUT WRAPPER: y shape: {y.shape}')
        if sample_weights is not None:
            sample_weights = format_sample_weights(sample_weights)
        return self.base.fit(x, y, sample_weights)

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        predicts the k-distances.

        :param x: numpy.ndarray, shape: (n, features), dtype: numpy.float64
            The data points. Do not contain k.

        :return: numpy.ndarray, shape: (n, ), dtype: numpy.float64
            The k_distances directly predicted by the base model.
        """
        y_pred = self.base.predict(x).astype(numpy.float32)
        return y_pred.reshape(-1).clip(min=0)

    def set_min_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ):
        if is_predicted:
            y_pred = x
        else:
            y_pred = self.predict(x)
        y_train = y.reshape(-1)
        # Point-wise prediction error
        diff = y_train - y_pred

        # Reshape to common form
        diff = diff.reshape(y.shape)

        if agg_point:
            # Compute minimal and maximal training error by aggregating over points
            self.min_diff = diff.min(axis=0)
            self.max_diff = diff.max(axis=0)
        else:
            # Compute minimal and maximal training error by aggregating over k
            self.min_diff = diff.min(axis=1)
            self.max_diff = diff.max(axis=1)

    def predict_bounds(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if is_predicted:
            y_pred = x
        else:
            y_pred = self.predict(x)

        pred_rs = np.reshape(y_pred, (-1, K_MAX))
        if agg_point:
            lower = pred_rs + self.min_diff[None, :]
            upper = pred_rs + self.max_diff[None, :]
        else:
            lower = pred_rs + self.min_diff[:, None]
            upper = pred_rs + self.max_diff[:, None]
        return lower.clip(min=0), upper.clip(min=0)


# Monotonicity Wrapper.
class MonotonicityWrapper(BoundModel, ABC):
    """
    A wrapper, that ensures monotonicity in the output, by using cumulative maximum/minimum for upper / lower bound.
    """

    def __init__(
        self,
        base: BoundModel,
    ):
        self.base = base

    def predict(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
    ) -> np.ndarray:
        lower, upper = self.predict_bounds(x, is_predicted)
        return 0.5 * (lower + upper)

    def predict_bounds(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = self.base.predict_bounds(x, is_predicted, agg_point)

        # Use cumulative maximum for lower bound
        lower = np.maximum.accumulate(lower, axis=1)

        # Use right-to-left cumulative minimum for upper bound
        upper = np.minimum.accumulate(upper[:, ::-1], axis=1)[:, ::-1]

        return lower, upper

    def set_min_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> None:
        self.base.set_min_max(x=x, y=y, is_predicted=is_predicted, agg_point=agg_point)


# Normalization Wrapper.
class NormalizationWrapper(BoundModel, ABC):
    """
    A wrapper, that wraps normalization around a BoundModel
    """

    def __init__(
        self,
        base: BoundModel,
        is_multi: bool = False,
    ):
        self.base = base
        self.min_diff = None
        self.max_diff = None
        self.is_multi = is_multi
        self.x_scaler = StandardScaler(with_mean=True, with_std=True)
        self.y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights=None
    ):
        x_norm = self.x_scaler.fit_transform(x)

        if not self.is_multi:
            y = y.reshape((-1, 1))

        y_norm = self.y_scaler.fit_transform(y)
        if not self.is_multi:
            y_norm = y_norm.reshape(-1)
        self.base.fit(x_norm, y_norm, sample_weights)

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x_norm = self.x_scaler.fit_transform(x)
        y_pred_norm = self.base.predict(x_norm)
        if not self.is_multi:
            y_pred_norm = y_pred_norm.reshape((-1, 1))
        pred = self.y_scaler.inverse_transform(y_pred_norm)
        return pred

    def set_min_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ):
        if is_predicted:
            y_pred = x
        else:
            y_pred = self.predict(x)
        y_train = y.reshape(-1)
        # Point-wise prediction error
        diff = y_train - y_pred

        # Reshape to common form
        diff = diff.reshape(y.shape)

        if agg_point:
            self.min_diff = diff.min(axis=0)
            self.max_diff = diff.max(axis=0)
        else:
            self.min_diff = diff.min(axis=1)
            self.max_diff = diff.max(axis=1)

    def predict_bounds(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if is_predicted:
            y_pred = x
        else:
            y_pred = self.predict(x)
        pred_rs = np.reshape(y_pred, (-1, K_MAX))
        if agg_point:
            lower = pred_rs + self.min_diff[None, :]
            upper = pred_rs + self.max_diff[None, :]
        else:
            lower = pred_rs + self.min_diff[:, None]
            upper = pred_rs + self.max_diff[:, None]
        return lower.clip(min=0), upper.clip(min=0)


def format_sample_weights(
    sample_weights: numpy.ndarray
) -> numpy.ndarray:
    """
        Depending on the Model Wrapper the sample weights have to be reshaped.
        For joint prediction each sample has just one input row and thus needs one sample weight.
        For this the mean of the sample weights is returned.

        :param sample_weights: numpy.ndarray, shape (n, k_max)
            The dataset
        :return: numpy.ndarray, shape (n)
            Return the formatted sample weights.
    """
    sample_weights = sample_weights.reshape(-1, K_MAX)
    return numpy.mean(sample_weights, axis=1).reshape(-1)


def format_data(
    coord,
    skd: Optional[numpy.ndarray] = numpy.empty(shape=0),
    k: Optional[int] = None,
) -> Union[Tuple[numpy.ndarray, numpy.ndarray], numpy.ndarray]:
    n, d = coord.shape
    if k is None:
        k = skd.shape[1]
    # Allocate full array
    new_x = numpy.empty(shape=(n, k, d + 1), dtype=numpy.float32)

    # First d entries are the coordinates (broadcast along second axis; the k-axis)
    new_x[:, :, :-1] = coord[:, None, :]

    # Last entry is k (broadcast along first and second axis)
    new_x[:, :, -1] = numpy.arange(k, dtype=numpy.float32)[None, None, :]

    x_rs = numpy.reshape(new_x, (-1, d + 1))

    if skd.any():
        y_rs = numpy.reshape(skd, (-1))
        return x_rs, y_rs
    else:
        return x_rs
