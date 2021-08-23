Module rpanda.datasets.common
=============================

Classes
-------

`NumpyTableDataset(X: numpy.ndarray, y: numpy.ndarray = None, shuffle: bool = True, batch_size: int = None)`
:   Iterates batches over Numpy objects.
    
    Args:
        X (np.ndarray): training/validation/test data. It is expected to be of shape (num_examples, num_features).
        y (np.ndarray, optional): training/validation/test labels. It is expected to be of shape (num_examples, 1). Defaults to None, in which case.
        shuffle (bool, optional): If True shuffles data ordering. Defaults to True.
        batch_size (int, optional): Size of the batch of data/labels to iterate. Defaults to None, in which case batch_size == len(X).
    
    Raises:
        ValueError: if X and y are not of same lenght it requires the user to redefine correcly input data.

    ### Instance variables

    `index: List[int]`
    :   Dataset index, i.e. array indices w.r.t. to input ordering after shuffle. 
        
        Returns:
            List[int]: dataset index.