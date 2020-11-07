"""Utility methods."""
import enum
import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import mlflow
import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import tqdm
from ray import tune
from scipy.spatial.qhull import ConvexHull
from torch.nn import functional

logger = logging.getLogger(name=__name__)


@torch.no_grad()
def all_knn(
    x: numpy.ndarray,
    k: int,
    batch_size: int,
    distance: str,
) -> numpy.ndarray:
    """
    Compute knn distance for all nodes using brute-force.

    :param x: shape: (n, d)
        The data points.
    :param k: >0
        The number of neighbors.
    :param batch_size: >0
        The batch size.
    :param distance:
        The distance to use.
    :return:
        The knn distances, shape: (n, k)
    """
    distance = get_distance_by_name(name=distance)
    device = resolve_device(device=None)
    logger.info(f"Using device {device}.")
    n = x.shape[0]
    distances = torch.empty(n, k, dtype=torch.float32, device="cpu")
    x = torch.as_tensor(data=x, dtype=torch.float32).to(device=device)
    for i in tqdm.trange(0, n, batch_size, unit_scale=True, unit="samples"):
        j = min(i + batch_size, n)
        distances[i:j, :] = distance.all_to_all(x=x[i:j], y=x).topk(k=k + 1, dim=-1, largest=False, sorted=True).values.cpu()[:, 1:]
    return distances.numpy()


def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


class MLFlowClient:
    def __init__(
        self,
        root: str,
        tracking_uri: str,
        experiment_name: str = 'nlkda'
    ):
        """
        Constructor.

        Connects to a running MLFlow instance in which the runs of the experiments are stored.
        Also creates an output root directory. The directory will be created if it does not exist.

        :param root: str
            The path of the output root.
        :param tracking_uri: str
            The uri where the MLFlow instance is running.
        :param experiment_name: str
            The name of the experiment on the MLFlow instance server.
        """
        mlflow.set_tracking_uri(uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id
        self.root = pathlib.Path(root)
        self.current_run = None

    def init_experiment(
        self,
        hyper_parameters: Dict[str, Any]
    ) -> Tuple[str, pathlib.Path]:
        """
        Initialise an experiment, i.e. a run in the specified experiment.
        Creates an entry describing the run's hyper-parameters, and associates an unique output directory.

        :param hyper_parameters: Dict
            The hyperparameters. Should be serialisable to JSON/BSON.
        :return: A tuple (id, path) where
            id:
                a unique ID of the experiment (equal to the ID in the MLFlow instance)
            path:
                the path to an existing directory for outputs of the experiment.
        """
        self.current_run = mlflow.start_run(experiment_id=self.experiment_id)

        # create output directory
        output_path = self.root / str(self.current_run.info.run_id)
        if output_path.is_dir():
            logging.error('Output path already exists! {p}'.format(p=output_path))
        output_path.mkdir(exist_ok=True, parents=True)
        hyper_parameters["out_dir"] = output_path

        mlflow.log_params(params=flatten_dict(hyper_parameters))
        # return id as experiment handle, and path of existing directory to store outputs to
        return self.current_run.info.run_id, output_path

    def finalise_experiment(
        self,
        result: Dict[str, Any],
    ) -> None:
        """
        Finalise an experiment by storing some (high-level) results into the experiment's metrics in the MLFlow run.

        :param result: Dict
            A flattened dictionary holding high-level results, e.g. a few numbers.

        :return: None.
        """
        mlflow.log_metrics(metrics=flatten_dict(result))
        mlflow.end_run()

    def unified_get_entries(
        self,
        keys: Sequence[str],
        equals: Sequence[bool],
        values: Sequence[str]
    ) -> pd.DataFrame:
        search_string = ''
        for index, key in enumerate(keys):
            operand = '=' if equals[index] else '!='
            tmp_str = f'{key} {operand} "{values[index]}"'
            if index != 0:
                tmp_str = f' and {tmp_str}'
            search_string = search_string + tmp_str
        return mlflow.search_runs(experiment_ids=self.experiment_id, filter_string=search_string)


def get_skyline(
    selection: pd.DataFrame,
    columns: List[str],
    smaller_is_better: Tuple[bool, ...],
):
    """
    Finds the skyline which contains all entries which are not dominated by another entry.

    :param columns: name of columns of the skyline
    :param selection: numpy.ndarray, shape: (n_eval, d), dtype: numpy.float
        A selection of entries which will be reduced to the skyline.
    :param smaller_is_better:
        Whether smaller or larger values are better.

    :return: The skyline.
    """
    # pylint: disable=unsubscriptable-object
    x = selection[columns].values.copy()

    # skyline candidates can only be found in the convex hull
    hull = ConvexHull(x)
    candidates = hull.vertices
    x = x[candidates]

    # turn sign for values where smaller is better => larger is always better
    x[:, smaller_is_better] *= -1

    # x[i, :] is dominated by x[j, :] if stronger(x[i, k], x[j, k]).all()
    # the skyline consists of points which are **not** dominated
    selected_candidates, = (~(x[:, None, :] < x[None, :, :]).all(axis=-1).any(axis=1)).nonzero()
    return selection.iloc[[candidates[i] for i in selected_candidates]].sort_values(by=columns)


def save_to_file(
    output_root: pathlib.Path,
    file_name: str,
    data: np.ndarray,
    overwrite: bool = False,
) -> pathlib.Path:
    """
    Save an array to a file.

    :param output_root:
        The directory.
    :param file_name:
        The file name.
    :param data:
        The data.
    :param overwrite:
        Whether to enforce overwriting existing files.

    :return:
        The path to the output file.
    """
    # ensure directory exists
    output_root.mkdir(exist_ok=True, parents=True)
    # compose file name
    file_path = output_root / file_name
    if file_path.is_file():
        logger.info(f"{file_path} exists.")
        if not overwrite:
            return file_path
    # save
    numpy.save(str(file_path), data)
    logger.info(f"Saved to {file_path}.")
    return file_path


def enum_values(enum_cls: Type[enum.Enum]) -> List:
    """Get a list of enum values."""
    return list(e.value for e in enum_cls)


class Distance:
    """Distance function for pytorch."""

    def all_to_all(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute pairwise distances.

        :param x: shape: (n, d)
            The left side.
        :param y: shape: (m, d)
            The right side.

        :return: shape: (m, n)
            The distance matrix, D[i, j] = dist(x_i, y_j)
        """
        raise NotImplementedError


class LpDist(Distance):
    """L_p distance."""

    def __init__(self, p: float = 2):
        self.p = p

    def all_to_all(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return torch.cdist(x, y, p=self.p, compute_mode='donot_use_mm_for_euclid_dist')


class CosineDistance(Distance):
    """Cosine distance."""

    def all_to_all(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        x_n, y_n = [functional.normalize(z, p=2, dim=-1) for z in (x, y)]
        return 1.0 - x_n @ y_n.t()


def get_distance_by_name(name: str) -> Distance:
    """Get a distance function by name."""
    if name == "minkowski":
        return LpDist()
    if name == "cosine":
        return CosineDistance()
    raise ValueError(name)


def resolve_device(device: Optional[torch.device]):
    """
    Resolve the torch device.

    :param device:
        The device.

    :return:
        If device is not None: the device
        Else: cuda if available, else cpu
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def tune_q_log_uniform(
    high: int,
    low: int = 1,
    q: int = 1,
):
    def func(spec):
        return int(max(low, numpy.round(numpy.random.uniform(numpy.log(low), numpy.log(high))))) // q * q

    return tune.sample_from(func)


def tune_enum(enum_cls: Type[enum.Enum]):
    return tune.choice(enum_values(enum_cls=enum_cls))


def tune_bool():
    return tune.choice([True, False])
