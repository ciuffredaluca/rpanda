import random
from typing import List, Tuple, Union
import numpy as np


class NumpyTableDataset(object):
    """Iterates batches over Numpy objects.

    Args:
        X (np.ndarray): training/validation/test data. It is expected to be of shape (num_examples, num_features).
        y (np.ndarray, optional): training/validation/test labels. It is expected to be of shape (num_examples, 1). Defaults to None, in which case.
        shuffle (bool, optional): If True shuffles data ordering. Defaults to True.
        batch_size (int, optional): Size of the batch of data/labels to iterate. Defaults to None, in which case batch_size == len(X).

    Raises:
        ValueError: if X and y are not of same lenght it requires the user to redefine correcly input data.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray = None,
                 shuffle: bool = True,
                 batch_size: int = None):

        # --- Collect data --- #
        assert len(X.shape) == 2
        if y is not None:
            assert len(y.shape) == 2
            assert y.shape[1] == 1
        self.__data = X
        self.__target = y
        if self.__target is not None:
            if len(self.__data) != len(self.__target):
                raise ValueError(
                    "Target and data are not same lenght! Please provide a consistent dataset.")

        self.num_features = self.__data.shape[1]

        # --- Data indexing --- #
        self.__index = list(range(self.__data.shape[0]))
        if shuffle:
            random.shuffle(self.__index)

        # --- Batches --- #
        self.batch_size = batch_size if batch_size is not None else self.__data.shape[0]
        self.num_batches = len(self.__index) // self.batch_size + \
            (len(self.__index) % self.batch_size > 0)

    def __len__(self) -> int:
        """Total number of items in the dataset.

        Returns:
            int: number of items in dataset.
        """
        return self.__data.shape[0]

    @property
    def index(self) -> List[int]:
        """Dataset index, i.e. array indices w.r.t. to input ordering after shuffle. 

        Returns:
            List[int]: dataset index.
        """
        return self.__index

    def __get_batch(self, idx: int, shuffle_at_epoch_end: bool = True) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Create dataset batch.

        Args:
            idx (int): batch index to extract.
            shuffle_at_epoch_end (bool, optional): Shuffle dataset index at each epoch. Defaults to True.

        Returns:
            Tuple[np.ndarray, Union[np.ndarray, None]]: Tuple (batch_X, batch_y). If labels are None, batch_y == None.
        """
        if idx > 0:
            idx = idx % (self.num_batches)
        if (idx == 0) and shuffle_at_epoch_end:
            random.shuffle(self.__index)
        lower = idx * self.batch_size
        upper = min((idx + 1) * self.batch_size, len(self.__index))
        if self.__target is None:
            return self.__data[self.__index[lower:upper], :], None
        else:
            return self.__data[self.__index[lower:upper], :], self.__target[self.__index[lower:upper], :]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        batch = self.__get_batch(self._idx)
        self._idx += 1
        return batch
