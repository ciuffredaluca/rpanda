from typing import Tuple
from rpanda.losses import L2NormLoss
import numpy as np
from rpanda.datasets import NumpyTableDataset
import logging

logger = logging.getLogger()


class LinearRegressor(object):
    """Linear regression

    This class implements a linear regressor, able to fit a given dataset using gradient descent and L2 loss.

    Args:
        dataset (NumpyTableDataset): Dataset.
        lr (float, optional): Learning rate. Defaults to 0.001.
    """

    def __init__(self, dataset: NumpyTableDataset, lr: float = 0.001):

        self.loss = L2NormLoss()
        self.ds = dataset
        self.theta = np.random.randn(*(1, self.ds.num_features)) * 0.001
        self.lr = lr

    def forward(self, X):
        return np.dot(X, self.theta.T).reshape(-1, 1)

    def __call__(self, X):
        return self.forward(X)

    def __update_parameters(self, gradients: np.ndarray):
        """Update model weights for gradient descent.

        Args:
            gradients (np.ndarray): gradients computed for a given batch at a given training step.
        """
        self.theta = self.theta - self.lr*gradients

    def training_step(self, batch: Tuple[np.ndarray, np.ndarray]) -> float:
        """Basic logic of a single training iteration.

        Args:
            batch (Tuple[np.ndarray, np.ndarray]): A tuple (batch_X, batch_y).

        Raises:
            ValueError: If no target value is provided it is required to redifine the dataset.
        Returns:
            float: Loss function at a given step.
        """

        X, y = batch
        if y is None:
            raise ValueError(
                "No target variable is provided: please review dataset definition.")
        preds = self.forward(X)
        error = preds - y
        loss = self.loss(y, preds)
        gradients = np.dot(X.T, (preds - y)).T / len(X)
        self.__update_parameters(gradients)
        return loss

    def fit(self, num_epochs: int = 10):
        """Perform model training.

        Args:
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        """
        batch_idx = 0
        epoch_idx = 0
        epoch_losses = []

        for batch in self.ds:
            step_loss = self.training_step(batch)
            epoch_losses.append(step_loss)
            if (batch_idx % self.ds.num_batches == 0) and batch_idx > 0:
                logger.info(
                    f"epoch: {epoch_idx}, loss: {np.mean(epoch_losses)}")
                epoch_losses = []
                epoch_idx += 1
            batch_idx += 1
            if epoch_idx == num_epochs:
                break
