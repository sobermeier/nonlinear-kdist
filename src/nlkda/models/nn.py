# coding=utf-8
"""Neural network with pytorch."""
import gc
import logging
import random as rn
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim

logger = logging.getLogger(__name__)


class Loss(nn.Module):
    def forward(
        self,
        y_pred: torch.FloatTensor,
        y_true: torch.FloatTensor,
        sample_weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        elementwise_loss = self._elementwise_loss(y_pred=y_pred, y_true=y_true)
        if sample_weights is not None:
            if len(elementwise_loss.size()) > 1:
                sample_weights = sample_weights.unsqueeze(-1)
            elementwise_loss = elementwise_loss * sample_weights.unsqueeze(-1)
        return elementwise_loss.mean()

    def _elementwise_loss(
        self,
        y_pred: torch.FloatTensor,
        y_true: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class MAELoss(Loss):
    def _elementwise_loss(
        self,
        y_pred: torch.FloatTensor,
        y_true: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return (y_pred - y_true).abs()


class MSELoss(Loss):
    def _elementwise_loss(
        self,
        y_pred: torch.FloatTensor,
        y_true: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return (y_pred - y_true) ** 2


class NeuralNetwork:
    """A neural network."""

    def __init__(
        self,
        units: int,
        loss: str,
        dropout: bool,
        dropout_rate: float,
        layers: int,
        batch_normalization: bool,
        batch_size: int = 1024,
        epochs: Optional[int] = 1000,
        input_shape: Optional[int] = 3,
        dense_units: Optional[int] = 1,
        x_scaler=None,
        y_scaler=None,
        is_normalized: bool = True,
        random_state: int = 0,
        patience: int = 3,
        min_delta: float = 1.0e-05,
        low_memory: bool = False,
    ):
        self.batch_size = batch_size
        # One epoch means using all data samples once to compute updates
        self.epochs = epochs
        self.input_shape = input_shape
        self.dense_units = dense_units
        self.layers = layers
        self.loss = loss
        if loss == 'mean_squared_error' or loss == "MSELoss()":
            self.loss_func = MSELoss()
        elif loss == 'mean_absolute_error' or loss == MAELoss():
            self.loss_func = MAELoss()
        else:
            raise ValueError(f'Unknown loss: "{loss}"')
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.units = units
        self.is_normalized = is_normalized
        self.random_state = random_state

        # early stopping
        self.patience = patience
        self.min_delta = min_delta

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f'----- device bla: {self.device}')
        logger.info(f'----- low mem: {low_memory}')

        np.random.seed(random_state)
        rn.seed(random_state)
        torch.random.manual_seed(seed=random_state)

        self.y_pred = None
        self.history = None

        self.low_memory = low_memory

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        if None in (self.y_scaler, self.x_scaler):
            self.x_scaler, self.y_scaler = self.init_scalers()
        self.model, self.optimizer = self.init_model()

    def init_scalers(self):
        x_scaler = StandardScaler(copy=not self.low_memory, with_mean=True, with_std=True)
        logger.info("x_scaler")
        y_scaler = MinMaxScaler(copy=not self.low_memory, feature_range=(0.1, 0.9))
        logger.info("y_scaler")
        return x_scaler, y_scaler

    def init_model(
        self,
    ):
        modules = []
        in_dim = self.input_shape
        use_bias = not self.batch_normalization
        for i in range(1, self.layers + 1):
            units = self.calc_units(i)
            modules.append(nn.Linear(in_features=in_dim, out_features=units, bias=use_bias))
            in_dim = units
            if self.batch_normalization:
                modules.append(nn.BatchNorm1d(num_features=in_dim))
            modules.append(nn.LeakyReLU())
            if self.dropout:
                modules.append(nn.Dropout(self.dropout_rate))

        # Layer layer.
        # - units should equal the number of output features
        # - activation sigmoid has a value range of (0, 1), i.e. restricting the output to a fixed range.
        # The target values should be preprocessed to have approx. the same range
        # - use_bias=True is used, as no batch normalisation follows.
        modules.append(nn.Linear(in_features=in_dim, out_features=self.dense_units, bias=True))
        if self.is_normalized:
            modules.append(nn.Sigmoid())
        else:
            modules.append(nn.LeakyReLU())

        model = nn.Sequential(*modules)
        print(model)

        # initialize weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Move model to device
        model = model.to(device=self.device)

        # initialize optimizer
        optimizer = optim.Adam(params=model.parameters())

        return model, optimizer

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        # Fit the model
        # verbose=2 -> print one line per epoch, no intra-epoch progress bar
        logger.info('fit y')

        if self.dense_units == 1:
            y = y.reshape((-1, 1))

        gc.collect()
        if self.is_normalized:
            x = self.x_scaler.fit_transform(x)
            y = self.y_scaler.fit_transform(y)
        logger.info('after fit transform')

        # move data to GPU
        x = torch.as_tensor(data=x, dtype=torch.float)
        y = torch.as_tensor(data=y, dtype=torch.float)
        if sample_weights is not None:
            assert sample_weights.shape[0] == x.shape[0]
            sample_weights = torch.as_tensor(data=sample_weights, dtype=torch.float)
        n = x.shape[0]

        self.history = []
        for epoch in range(self.epochs):
            # Set model into training mode
            self.model.train()

            train_epoch_loss = 0.
            for i in range(0, n, self.batch_size):
                # randomly sample batch: while this does not guarantee that each sample is seen *exactly* once per
                # epoch, it guarantees it on average
                ind = torch.randint(n, size=(self.batch_size,))
                j = min(i + self.batch_size, n)

                # forward pass
                batch_x = x[ind].to(self.device)

                batch_pred_y = self.model.forward(batch_x)

                # compute loss
                batch_y = y[ind].to(self.device)
                if sample_weights is not None:
                    batch_sample_weights = sample_weights[ind].to(self.device)
                else:
                    batch_sample_weights = None

                # logger.info(f"------- Loss func: {self.loss_func}")
                loss_value = self.loss_func.forward(y_pred=batch_pred_y, y_true=batch_y,
                                                    sample_weights=batch_sample_weights)

                # torch *accumulates* the gradient; hence we need to zero it before computing new gradients
                self.optimizer.zero_grad()

                # compute gradients
                loss_value.backward()

                # update parameters
                self.optimizer.step()

                # Accumulate loss in epoch loss
                real_batch_size = (j - i)
                train_epoch_loss += loss_value.item() * real_batch_size

            # Epoch loss = average over all samples
            train_epoch_loss /= n
            self.history.append(train_epoch_loss)
            logger.info(f'Epoch {epoch:5}: {train_epoch_loss:2.7f}')

            # Early stopping
            if len(self.history) > self.patience:
                if train_epoch_loss >= (1 - self.min_delta) * max(self.history[-self.patience:]):
                    logger.info('Early stopping')
                    break
        if self.low_memory and self.is_normalized:
            self.x_scaler.inverse_transform(x)
            self.y_scaler.inverse_transform(y)

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        # set into evaluation mode
        self.model.eval()

        # do not track gradients (leads to faster code)
        with torch.no_grad():
            # Predict values
            if self.is_normalized:
                # De-normalise again
                y_pred = self.y_scaler.inverse_transform(
                    self.model.forward(
                        torch.as_tensor(
                            data=self.x_scaler.fit_transform(x),
                            dtype=torch.float,
                            device=self.device,
                        )).cpu().numpy()
                )
                if self.low_memory:
                    self.x_scaler.inverse_transform(x)
            else:
                y_pred = self.model.forward(
                    torch.as_tensor(
                        data=x,
                        dtype=torch.float,
                        device=self.device,
                    )).cpu().numpy()
        return y_pred

    def calc_units(
        self,
        layer: int,
    ):
        return self.units * (2 ** (layer - 1))

    def __reduce__(self):
        """
            This function reduces the object attributes.
            This is needed for saving the object with pickle because self.history can not be pickled
            and therefore has to be ignored
        """
        return self.__class__, (
            self.units,
            self.loss,
            self.dropout,
            self.dropout_rate,
            self.layers,
            self.batch_normalization,
            self.batch_size,
            self.epochs,
            self.input_shape,
            self.dense_units,
            self.x_scaler,
            self.y_scaler,
            self.is_normalized,
            self.random_state,
            self.patience,
            self.min_delta
        )
