# coding=utf-8
"""
Evaluation utilities.
"""
import logging
import pathlib
from typing import Optional, Tuple, Union

import numpy
import numpy as np
import torch
import tqdm
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from torch import nn

from .models.base import BoundModel, MonotonicityWrapper, NormalizationWrapper
from .models.nn import NeuralNetwork
from .models.utils import get_candidate_file_name
from .utils import get_distance_by_name, resolve_device, save_to_file


def get_candidate_set_size(
    lower: numpy.ndarray,
    upper: numpy.ndarray,
    x: numpy.ndarray,
    distance: str,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> numpy.ndarray:
    """
    Brute force alternative for computing the candidate set size using pytorch.

    Also computes the distances.

    :param lower: shape: (n, k), dtype: float
        The lower bounds (one per data point)
    :param upper: shape: (n, k), dtype: float
        The upper bounds (one per data point)
    :param x: shape: (n, d)
        The data points.
    :param distance:
        The distance function.
    :param batch_size:
        Optional batch size for evaluation to mitigate out-of-memory errors.
    :param device:
        The device to use.

    :return: shape: (n, k), dtype: int
        The candidate set sizes per data point.
    """
    device = resolve_device(device=device)
    logging.info(f'Using device={device}')

    distance = get_distance_by_name(name=distance)

    # convert to tensors
    lower = torch.as_tensor(data=lower, dtype=torch.float)
    upper = torch.as_tensor(data=upper, dtype=torch.float)
    x = torch.as_tensor(data=x, dtype=torch.float, device=device)

    n, k = lower.shape
    if batch_size is None:
        batch_size = n

    num_candidates = numpy.empty(shape=(n, k), dtype=numpy.int32)
    with torch.no_grad():
        while batch_size > 0:
            try:
                for start in tqdm.trange(0, n, batch_size, unit='sample', unit_scale=True, leave=False,
                                         desc=f'Evaluation (device={device}, batch_size={batch_size})'):
                    end = min(start + batch_size, n)
                    real_dist_batch = distance.all_to_all(x=x[start:end], y=x)
                    num_candidates[start:end, :] = (
                        (lower[start:end, :, None].to(device=device) <= real_dist_batch[:, None, :]).sum(dim=-1) - (
                        real_dist_batch[:, None, :] > upper[start:end, :, None].to(device=device)).sum(dim=-1)
                    ).cpu().numpy()
                    del real_dist_batch  # release memory
                logging.info('Finished with batch_size=%d', batch_size)
                break
            except RuntimeError as error:
                logging.info('error: %s', error)
                torch.cuda.empty_cache()
                batch_size //= 2
        if batch_size == 0:
            raise RuntimeError('Could not evaluate even with batch_size=1.')

    print(f'num_candidates: {num_candidates}')
    return num_candidates


def get_model_size(model: Union[DecisionTreeRegressor, nn.Module, NormalizationWrapper]) -> int:
    """
    Calculate the number of parameters of a model.

    :param model: DecisionTreeRegressor | nn.Module
        The model

    :return: int
        The number of parameters needed by the model.
    """
    if isinstance(model, DecisionTreeRegressor):
        tree = model.tree_
        node_count = tree.node_count
        node_count_leaves = (node_count + 1) // 2
        node_count_inner = node_count - node_count_leaves
        _, n_outputs, _ = tree.value.shape

        # inner nodes store: threshold (float), attribute (int), 2*children (int)
        # leaf nodes store: #outputs values (float)

        n_parameters = node_count_leaves * n_outputs + node_count_inner * 4
    elif isinstance(model, RandomForestRegressor):
        n_parameters = sum(get_model_size(sub_model) for sub_model in model.estimators_)
    elif isinstance(model, nn.Module):
        n_parameters = sum(p.numel() for layer in model.children() for p in layer.parameters() if
                           not isinstance(layer, nn.modules.batchnorm.BatchNorm1d))
    elif isinstance(model, NeuralNetwork):
        n_parameters = get_model_size(model.model)
    elif isinstance(model, MultiOutputRegressor):
        n_parameters = sum(get_model_size(sub_model) for sub_model in model.estimators_)
    elif isinstance(model, GradientBoostingRegressor):
        n_parameters = sum(get_model_size(sub_model) for sub_model in model.estimators_[:, 0])
    elif isinstance(model, AdaBoostRegressor):
        n_parameters = model.n_estimators + sum(get_model_size(sub_model) for sub_model in model.estimators_)
    elif isinstance(model, MLPRegressor):
        n_parameters = sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_)
    elif isinstance(model, NormalizationWrapper):
        # NORMALIZATION DOES NOT HAVE ANY EFFECT IN OUR SETUP, SO IGNORE NORMALIZATION
        n_parameters = get_model_size(model.base)
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    return n_parameters


def evaluate_model(
    x: np.ndarray,
    distance: str,
    eval_batch_size: int,
    model: BoundModel,
    output_path: pathlib.Path,
    pred: np.ndarray,
    skd_max: np.ndarray,
    skd_min: np.ndarray,
    agg_point: bool = False,
    monotonous: bool = False,
    both: bool = False,
    kd=None
) -> Tuple[float, float]:
    if monotonous:
        model = MonotonicityWrapper(base=model)

    if both:
        model.set_min_max(x=pred, y=kd, is_predicted=True, agg_point=True)
        lower_p, upper_p = model.predict_bounds(x=pred, is_predicted=True, agg_point=True)
        lower_p = lower_p.clip(min=skd_min, max=skd_max)
        upper_p = upper_p.clip(min=skd_min, max=skd_max)
        assert (lower_p <= upper_p).all()

        model.set_min_max(x=pred, y=kd, is_predicted=True, agg_point=False)
        lower_k, upper_k = model.predict_bounds(x=pred, is_predicted=True, agg_point=False)
        lower_k = lower_k.clip(min=skd_min, max=skd_max)
        upper_k = upper_k.clip(min=skd_min, max=skd_max)
        assert (lower_k <= upper_k).all()

        lower = np.maximum(lower_k, lower_p)
        upper = np.minimum(upper_k, upper_p)

    else:
        lower, upper = model.predict_bounds(x=pred, is_predicted=True, agg_point=agg_point)
        lower = lower.clip(min=skd_min, max=skd_max)
        upper = upper.clip(min=skd_min, max=skd_max)
        assert (lower <= upper).all()

    # select device
    cs = get_candidate_set_size(
        lower=lower,
        upper=upper,
        x=x,
        distance=distance,
        batch_size=eval_batch_size,
    )
    save_to_file(
        output_root=output_path,
        file_name=get_candidate_file_name(monotonous=monotonous, agg_point=agg_point, both=both),
        data=cs,
    )
    cs_mean = np.mean(cs)
    cs_median = np.median(cs)
    return cs_mean, cs_median
