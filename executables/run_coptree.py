"""Evaluate the MRkNNCoP tree model."""
import argparse
import pathlib

import numpy as np

from nlkda.data.base import DatasetEnum, get_dataset_size
from nlkda.data.loader import get_data
from nlkda.eval import get_candidate_set_size
from nlkda.models.linear import MRkNNCoPTreeBounds
from nlkda.models.utils import ModelEnum, get_candidate_file_name
from nlkda.settings import K_MAX
from nlkda.utils import MLFlowClient, enum_values, flatten_dict, save_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DatasetEnum.OL.value, help="The name of the dataset.",
                        choices=enum_values(enum_cls=DatasetEnum))
    parser.add_argument("--data_root", type=str, default="/tmp/data", help="The directory where data is stored.")
    parser.add_argument("--tracking_uri", type=str, default="http://localhost:5000", help="The MLFlow tracking URI.")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().absolute()

    ds_enum = DatasetEnum(args.dataset)
    db = MLFlowClient(root=data_root / "experiments", tracking_uri=args.tracking_uri)

    experiment_parameters = {
        "data_root": str(data_root),
        "tracking_uri": args.tracking_uri,
        "model": {"model_type": ModelEnum.COP.value},
        "dataset": ds_enum.value,
        "k": K_MAX,
        "n": get_dataset_size(ds_enum),
    }
    experiment_parameters = flatten_dict(experiment_parameters)
    run_id, output_path = db.init_experiment(hyper_parameters=experiment_parameters)
    output_path = pathlib.Path(output_path)

    # load data
    x, y, distance = get_data(dataset_enum=ds_enum, data_root=data_root)

    # fit model
    cop_model = MRkNNCoPTreeBounds()
    cop_model.fit(x=x, y=y, sample_weights=None)

    # inference
    # cop_model.predict(x=x)
    lower, upper = cop_model.predict_bounds(x=x)

    # evaluate
    cs_cop = get_candidate_set_size(
        lower=lower,
        upper=upper,
        x=x,
        distance=distance,
        batch_size=30,
    )

    size = len(x) * 4
    result = {
        "cs_median": np.median(cs_cop),
        "cs_mean": np.mean(cs_cop),
        "model_size": size
    }
    db.finalise_experiment(result=result)

    save_to_file(
        output_root=output_path,
        file_name=get_candidate_file_name(monotonous=False, agg_point=False),
        data=cs_cop,
    )


if __name__ == '__main__':
    main()
