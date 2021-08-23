Module rpanda.models.regression
===============================

Classes
-------

`LinearRegressor(dataset: rpanda.datasets.common.NumpyTableDataset, lr: float = 0.001)`
:   Linear regression
    
    This class implements a linear regressor, able to fit a given dataset using gradient descent and L2 loss.
    
    Args:
        dataset (NumpyTableDataset): Dataset.
        lr (float, optional): Learning rate. Defaults to 0.001.

    ### Methods

    `fit(self, num_epochs: int = 10)`
    :   Perform model training.
        
        Args:
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.

    `forward(self, X)`
    :

    `training_step(self, batch: Tuple[numpy.ndarray, numpy.ndarray]) ‑> float`
    :   Basic logic of a single training iteration.
        
        Args:
            batch (Tuple[np.ndarray, np.ndarray]): A tuple (batch_X, batch_y).
        
        Raises:
            ValueError: If no target value is provided it is required to redifine the dataset.
        Returns:
            float: Loss function at a given step.