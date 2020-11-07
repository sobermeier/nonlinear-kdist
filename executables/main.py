"""HPO script."""
import argparse
import logging
import pathlib
import random as rn
import sys
from enum import Enum
from typing import Any, Mapping, Tuple

import numpy as np
import torch
from ray import tune
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nlkda.data.base import DatasetEnum, get_dataset_size
from nlkda.data.loader import get_data
from nlkda.eval import evaluate_model, get_model_size
from nlkda.models.base import BoundModel, FormulationWrapperEnum, is_multi_output
from nlkda.models.boosting import BoostingWithCsModel
from nlkda.models.utils import ModelEnum, configure_param_space_nn, create_model_from_config, \
    create_wrapper_from_config, save_model_to_directory
from nlkda.settings import K_MAX
from nlkda.utils import MLFlowClient, enum_values, flatten_dict, save_to_file, tune_bool, tune_enum, tune_q_log_uniform


def _sample_max_leaf_nodes(spec) -> int:
    n = spec.config.n
    k = spec.config.k
    return tune_q_log_uniform(low=1, high=n * k, q=1)


def get_model_search_space(model_type: ModelEnum) -> Mapping[str, Any]:
    if model_type == ModelEnum.RF:
        return dict(
            max_depth=tune.randint(1, 10),
            n_estimators=tune_q_log_uniform(high=100, q=1),
        )
    elif model_type == ModelEnum.DT:
        return dict(
            max_leaf_nodes=tune.sample_from(_sample_max_leaf_nodes),
        )
    elif model_type == ModelEnum.ADB:
        return dict(
            n_estimators=tune_q_log_uniform(low=1, high=500, q=1),
            learning_rate=tune.loguniform(1.0e-04, 1.0e+01),
        )
    elif model_type == ModelEnum.GB:
        return dict(
            max_leaf_nodes=tune_q_log_uniform(low=4, high=15, q=1),
            n_estimators=tune_q_log_uniform(high=500, q=1),
            learning_rate=tune.loguniform(1.0e-04, 1.0e+01),
        )
    elif model_type == ModelEnum.NN:
        return dict(
            units=tune.randint(10, 28),
            layers=tune.randint(2, 9),
            dropout=tune_bool(),
            dropout_rate=tune.uniform(0.1, 0.5),
            batch_size=tune.choice([2 ** i for i in range(6, 10)]),
            loss=tune.choice(['mean_squared_error', 'mean_absolute_error']),
            batch_normalization=tune_bool(),
            is_normalized=tune_bool(),
            patience=4,
        )
    elif model_type == ModelEnum.KN:
        return dict(
            n_neighbors=1,
        )
    elif model_type == ModelEnum.COP:
        return dict()
    else:
        raise ValueError(model_type)


def _prepare_logger():
    # setup logger
    logger_main = logging.getLogger(__name__)
    logger_main.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("\n%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    # add formatter to ch
    sh.setFormatter(formatter)
    logger_main.addHandler(sh)
    logger_main.propagate = False
    return logger_main


def objective(config, reporter):
    """The optimization objective."""
    logger_main = _prepare_logger()

    # setup random seed
    random_state = 0
    np.random.seed(random_state)
    rn.seed(random_state)
    torch.random.manual_seed(seed=random_state)

    # get data
    data_root = pathlib.Path(config["data_root"])
    x, y, distance = get_data(
        dataset_enum=DatasetEnum(config["dataset"]),
        data_root=data_root,
    )
    n_samples, dimensions = x.shape
    skd_max = np.max(y, axis=0)
    skd_min = np.min(y, axis=0)

    # SAMPLE WEIGHTS
    sw_type = config["sample_weights"]
    if sw_type == SampleWeightsEnum.MEAN_CS_K.value:
        boost_iterations = config["boosting"]["iterations"]
    else:
        boost_iterations = 1

    if config["model"]["model_type"] == ModelEnum.NN.value:
        config = configure_param_space_nn(config, dimensions)

    config["clipped"] = True

    db = MLFlowClient(root=data_root / "experiments", tracking_uri=config["tracking_uri"])

    try:
        # create base model like RandomForest or GradientBoost etc.
        model_obj = create_model_from_config(
            model=ModelEnum(config["model"]["model_type"]),
            params=config["model"]["params"],
            is_multi=is_multi_output(FormulationWrapperEnum(config["model"]["formulation"]))
        )

        # create model wrapper like KAsInputWrapper or DiffKOutputWrapper etc.
        model = create_wrapper_from_config(
            model_wrapper=FormulationWrapperEnum(config["model"]["formulation"]),
            params={"base": model_obj}
        )

        # Fit model
        if boost_iterations > 1:
            sw_agg_point = False if sw_type == SampleWeightsEnum.MEAN_CS_K.value else True
            logger_main.info("Creating Boosting Model for CS!")
            boosting_model = BoostingWithCsModel(
                base=model,
                dataset=config["dataset"],
                sw_agg_point=sw_agg_point,
                iterations=boost_iterations,
            )
            boosting_model.fit(x=x, y=y, sample_weights=None)
            model = boosting_model.base
        else:
            model.fit(x=x, y=y, sample_weights=None)

        # create experiment
        experiment_parameters = flatten_dict(config)
        run_id, output_path = db.init_experiment(hyper_parameters=experiment_parameters)
        output_path = pathlib.Path(output_path)

        # save fitted model to output path
        save_model_to_directory(directory=output_path, model=model)

        # evaluate
        eval_batch_size = 30

        pred = model.predict(x=x)
        save_to_file(
            output_root=output_path,
            file_name="pred_k_dist",
            data=pred,
        )

        # ..... MODEL_SIZE
        model_size = get_model_size(model.base)
        # ..... MAE , MSE
        mae = mean_absolute_error(y.reshape(-1), pred)
        mse = mean_squared_error(y.reshape(-1), pred)

        # error over points --> O(k_max), model size increases by 2*k_max
        min_error, max_error, cs_mean_p, cs_median_p, cs_mean_mono_p, cs_median_mono_p = _evaluate_aggregation(
            x=x,
            kd=y,
            distance=distance,
            eval_batch_size=eval_batch_size,
            model=model,
            output_path=output_path,
            pred=pred,
            skd_max=skd_max,
            skd_min=skd_min,
            agg_point=True,
        )
        # error over k --> O(n), model size increases by 2*n
        _, _, cs_mean_k, cs_median_k, cs_mean_mono_k, cs_median_mono_k = _evaluate_aggregation(
            x=x,
            kd=y,
            distance=distance,
            eval_batch_size=eval_batch_size,
            model=model,
            output_path=output_path,
            pred=pred,
            skd_max=skd_max,
            skd_min=skd_min,
            agg_point=False,
        )

        # combine error over points and k, model size increases by 2*n + 2*k_max
        _, _, cs_mean_comb, cs_median_comb, cs_mean_mono_comb, cs_median_mono_comb = _evaluate_aggregation(
            x=x,
            kd=y,
            distance=distance,
            eval_batch_size=eval_batch_size,
            model=model,
            output_path=output_path,
            pred=pred,
            skd_max=skd_max,
            skd_min=skd_min,
            both=True
        )

        # finalise experiment
        result = {
            "mae": mae,
            "mse": mse,
            "model_size": model_size,
            "size_agg_p": model_size + (2 * K_MAX),
            "size_agg_k": model_size + (2 * config["n"]),
            "size_combined": model_size + (2 * K_MAX) + (2 * config["n"]),
            "max_error": {
                "error": max_error,
                "k": int(np.argmax(model.max_diff)) + 1,
            },
            "min_error": {
                "error": min_error,
                "k": int(np.argmin(model.min_diff)) + 1,
            },
            "cs_mean_agg_p": cs_mean_p,
            "cs_median_agg_p": cs_median_p,
            "cs_mean_mono_agg_p": cs_mean_mono_p,
            "cs_median_mono_agg_p": cs_median_mono_p,
            "cs_mean_agg_k": cs_mean_k,
            "cs_median_agg_k": cs_median_k,
            "cs_mean_mono_agg_k": cs_mean_mono_k,
            "cs_median_mono_agg_k": cs_median_mono_k,
            "cs_mean_combined": cs_mean_comb,
            "cs_median_combined": cs_median_comb,
            "cs_mean_mono_combined": cs_mean_mono_comb,
            "cs_median_mono_combined": cs_median_mono_comb
        }

        # log all results
        result = flatten_dict(result)
        db.finalise_experiment(result=result)
        reporter(cs_mean_mono_p=cs_mean_mono_p)
        return result

    except RuntimeError as error:
        logger_main.warning(error)
        logger_main.warning("Oops!", sys.exc_info()[0], "occured.")
        logger_main.warning(sys.exc_info()[1], "  : value")
        return


def _evaluate_aggregation(
    x: np.ndarray,
    kd: np.ndarray,
    distance: str,
    eval_batch_size: int,
    model: BoundModel,
    output_path: pathlib.Path,
    pred: np.ndarray,
    skd_max: np.ndarray,
    skd_min: np.ndarray,
    agg_point: bool = False,
    both: bool = False
) -> Tuple[float, float, float, float, float, float]:
    model.set_min_max(x=pred, y=kd, is_predicted=True, agg_point=agg_point)
    max_error = max(model.max_diff)
    min_error = min(model.min_diff)
    result = tuple()
    for monotonous in (False, True):
        result = result + evaluate_model(
            x=x,
            distance=distance,
            eval_batch_size=eval_batch_size,
            model=model,
            output_path=output_path,
            pred=pred,
            skd_max=skd_max,
            skd_min=skd_min,
            agg_point=agg_point,
            monotonous=monotonous,
            both=both,
            kd=kd
        )
    return (min_error, max_error) + result


class SampleWeightsEnum(Enum):
    """Enum for sample weights."""
    NONE = "NO"
    MEAN_CS_K = "mean_cs_agg_k"


def main(
    model_type: ModelEnum,
    dataset: DatasetEnum,
    s_w: SampleWeightsEnum,
    tracking_uri: str,
    data_root: pathlib.Path = pathlib.Path("/mnt/data"),
    local_dir: pathlib.Path = pathlib.Path("~"),
    num_samples: int = 10,
) -> None:
    """The main HPO routine, using ray tune."""
    n_gpus = 1 if torch.cuda.is_available() else 0
    analysis = tune.run(
        objective,
        config=dict(
            data_root=str(data_root),
            tracking_uri=tracking_uri,
            dataset=dataset.value,
            k=K_MAX,
            n=get_dataset_size(dataset),
            model=dict(
                model_type=model_type.value,
                formulation=tune_enum(enum_cls=FormulationWrapperEnum),
                params=get_model_search_space(model_type),
            ),
            sample_weights=s_w.value,
            boosting=dict(
                iterations=3,
            )
        ),
        local_dir=str(local_dir.expanduser().absolute()),
        num_samples=num_samples,
        resources_per_trial={
            "cpu": 3,
            "gpu": n_gpus
        },
    )

    print("Best config: ", analysis.get_best_config(metric="cs_mean_mono_p", mode="min"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DatasetEnum.OL.value, help="The name of the dataset.",
                        choices=enum_values(enum_cls=DatasetEnum))
    parser.add_argument("--model", type=str, default=ModelEnum.NN.value, help="The name of the model.",
                        choices=enum_values(enum_cls=ModelEnum))
    parser.add_argument("--sample_weight", type=str, default=SampleWeightsEnum.NONE.value,
                        help="The name of the sample weights.", choices=enum_values(enum_cls=SampleWeightsEnum))
    parser.add_argument("--data_root", type=str, default="/tmp/data", help="The directory where data is stored.")
    parser.add_argument("--tracking_uri", type=str, default="http://localhost:5000", help="The MLFlow tracking URI.")
    parser.add_argument("--trials", type=int, default=10, help="The number of HPO trials to run.")
    args = parser.parse_args()

    main(
        model_type=ModelEnum(args.model),
        dataset=DatasetEnum(args.dataset),
        s_w=SampleWeightsEnum(args.sample_weight),
        data_root=pathlib.Path(args.data_root),
        tracking_uri=args.tracking_uri,
        num_samples=args.trials,
    )
