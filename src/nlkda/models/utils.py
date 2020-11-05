"""Model utilities for storing/loading models and configuring search spaces."""
import logging
import pathlib
import pickle
from enum import Enum
from typing import Any, Dict, Optional

import torch
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from .base import BoundModel, FormulationWrapperEnum, wrapper_options
from .linear import MRkNNCoPTreeBounds
from .nn import NeuralNetwork
from ..settings import K_MAX

logger = logging.getLogger(name=__name__)


class ModelEnum(Enum):
    """An enum for model classes."""
    DT = "decision_tree"
    RF = "random_forest"
    ADB = "ada_boost"
    COP = "cop_tree"
    GB = "gradient_boost"
    NN = "neural_network"
    KN = "neighbors"


def _decision_tree(
    params,
):
    return DecisionTreeRegressor(**params["model_params"])


def _random_forest(
    params,
):
    return RandomForestRegressor(**params["model_params"])


def _ada_boost(
    params,
):
    m = AdaBoostRegressor(**params["model_params"])
    if params["is_multi"]:
        return MultiOutputRegressor(estimator=m)
    else:
        return m


def _cop_tree(
    params
):
    return MRkNNCoPTreeBounds()


def _g_boost(
    params
):
    m = GradientBoostingRegressor(**params["model_params"])
    if params["is_multi"]:
        return MultiOutputRegressor(estimator=m)
    else:
        return m


def _neural_network(
    params
):
    return NeuralNetwork(**params["model_params"])


def _neighbors(
    params
):
    return KNeighborsRegressor(**params["model_params"])


# map the inputs to the function blocks
factories = {
    ModelEnum.RF: _random_forest,
    ModelEnum.DT: _decision_tree,
    ModelEnum.ADB: _ada_boost,
    ModelEnum.COP: _cop_tree,
    ModelEnum.GB: _g_boost,
    ModelEnum.NN: _neural_network,
    ModelEnum.KN: _neighbors,
}


def is_wrapper_single_prediction(fw: FormulationWrapperEnum):
    return fw == FormulationWrapperEnum.K_AS_INPUT.value


def configure_param_space_nn(param_space, dimensions):
    if is_wrapper_single_prediction(param_space["model"]["formulation"]):
        add_params = {
            'input_shape': dimensions + 1,
            'dense_units': 1
        }
    else:
        add_params = {
            'input_shape': dimensions,
            'dense_units': K_MAX
        }
    param_space["model"]["params"] = {**add_params, **param_space["model"]["params"]}
    return param_space


def get_candidate_file_name(
    monotonous: bool,
    agg_point: bool,
    both: bool = False,
) -> str:
    """Get canonical filename for candidate set sizes."""
    if both:
        return f"candidate_sizes{'_mono' if monotonous else ''}_k_point"
    else:
        return f"candidate_sizes{'_mono' if monotonous else ''}_{'_point' if agg_point else '_k'}"


def create_model_from_config(
    model: ModelEnum,
    params: Optional[Dict[str, Any]] = None,
    is_multi: Optional[bool] = False,
) -> BoundModel:
    """Create a model from a configuration."""
    params = params or dict()
    params = {"model_params": params, "is_multi": is_multi}
    return factories[model](params=params)


def create_wrapper_from_config(
    model_wrapper: FormulationWrapperEnum,
    params: Optional[Dict[str, Any]] = None,
) -> BoundModel:
    """Create a wrapper from a configuration."""
    params = params or dict()
    return wrapper_options[model_wrapper](params=params)


def save_model_to_directory(
    directory: pathlib.Path,
    model: BoundModel,
) -> None:
    """Save model to a directory."""
    directory.mkdir(exist_ok=True, parents=True)
    if isinstance(model.base, NeuralNetwork):
        torch.save(model.base.model, directory / "model.pth")
        with (directory / "wrapper.pkl").open("wb") as file:
            pickle.dump(model, file)
    else:
        with (directory / "model.pkl").open("wb") as file:
            pickle.dump(model, file)


def load_model_from_directory(
    directory: pathlib.Path,
    map_device: torch.device = torch.device('cpu'),
) -> BoundModel:
    """Load a model from a directory."""
    simple_model_path = directory / "model.pkl"
    torch_model_path = directory / "model.pth"
    if simple_model_path.is_file():
        with simple_model_path.open("rb") as file:
            model = pickle.load(file)
    elif torch_model_path.is_file():
        # load sequential
        nnt_model = torch.load(torch_model_path, map_location=map_device)
        nnt_model.eval()
        # load wrapper
        with (directory / "wrapper.pkl").open("rb") as file:
            model = pickle.load(file)
        model.base.model = nnt_model
    else:
        raise ValueError(f'Could not file model at {directory}')
    return model
