import numpy as np
import logging

logger = logging.getLogger()


class L2NormLoss(object):
    """L2-norm loss

    Computes squared error between a target and a prediction array.

    Args:
        object ([type]): [description]
    """

    def __call__(self, target: np.ndarray, preds: np.ndarray) -> float:
        assert target.shape == preds.shape, "target and preds should have same shape."
        loss = np.mean((target - preds)**2)
        if np.isnan(loss):
            logger.warning("Loss has reached NaN value.")

        return loss
