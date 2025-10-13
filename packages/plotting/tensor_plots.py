
import matplotlib.pyplot as plt
import numpy as np


def plot_dimension_distribuitions(tensor: np.ndarray , dim:int = None, nplots:int = None, dim_labels: list = None) -> None:
    """
    Grid subplots for n random indices in the dimension dim of a ndarray.
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.numpy()
        except (AttributeError, TypeError):
            try:
                tensor = np.array(tensor)
            except Exception:
                raise ValueError("Input tensor must be a numpy array or convertible to one.")
    
    if dim is None:
        dim = 0  # first dimension by default
    if nplots is None:
        nplots = min(6, tensor.shape[dim])  # up to 6 plots by default
    if nplots > tensor.shape[dim]:
        nplots = tensor.shape[dim]
    if dim_labels is None:
        dim_labels = [f"Dim {i}" for i in range(tensor.ndim)]
    if len(dim_labels) != tensor.ndim:
        raise ValueError("dim_labels length must match tensor dimensions.")
    
    random_indices = np.random.choice(tensor.shape[dim], size=nplots, replace=False)
    print(nplots, dim, dim_labels)
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.ceil(nplots / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nplots):
        axes[i//ncols, i%ncols].hist(tensor.take(random_indices[i], axis=dim).ravel(), bins=1000, color='blue', alpha=0.7)
        axes[i//ncols, i%ncols].set_title(f"Distribution of {dim_labels[dim]} (index: {random_indices[i]})")
        axes[i//ncols, i%ncols].set_xlabel(f"{dim_labels[dim]} dimension values")
        axes[i//ncols, i%ncols].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
