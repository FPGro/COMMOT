import numpy as np
from scipy import sparse


def to_dense(X):
    """Convert sparse or array-like to dense numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)