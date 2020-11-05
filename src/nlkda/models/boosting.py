from typing import Tuple

import numpy as np
import torch

from .base import BoundModel
from ..data.base import DatasetEnum
from ..eval import get_candidate_set_size


class BoostingWithCsModel(BoundModel):

    def __init__(
        self,
        iterations: int,
        dataset: str,
        base: BoundModel,
        sw_agg_point: bool
    ):
        self.computing_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.iterations = iterations
        self.dataset = DatasetEnum(dataset)
        self.distance = "minkowski" if self.dataset != DatasetEnum.WE_EN else "cosine"
        self.base = base
        self.sw_agg_point = sw_agg_point

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None
    ):
        for i in range(self.iterations):
            print(f'boosting iteration {i}')
            self.base.fit(x=x, y=y, sample_weights=sample_weights)
            if i < self.iterations - 1:
                pred = self.base.predict(x=x)
                self.base.set_min_max(x=pred, y=y, is_predicted=True, agg_point=self.sw_agg_point)

                lower, upper = self.base.predict_bounds(x=pred, is_predicted=True, agg_point=self.sw_agg_point)
                assert (lower <= upper).all()
                # ---- cs brute force
                cs = get_candidate_set_size(lower=lower, upper=upper, x=x, distance=self.distance,
                                            batch_size=30, device=self.computing_device)
                sample_weights = cs.reshape(-1)

    def predict(
        self,
        x: np.ndarray
    ):
        return self.base.predict(x=x)

    def set_min_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = False
    ):
        self.base.set_min_max(x=x, y=y, is_predicted=is_predicted, agg_point=agg_point)

    def predict_bounds(
        self,
        x: np.ndarray,
        is_predicted: bool = False,
        agg_point: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.base.predict_bounds(x=x, is_predicted=is_predicted, agg_point=agg_point)
