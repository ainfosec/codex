from typing import Optional

import numpy as np
from torch import Tensor


def umap_embed(
    vectors: Tensor,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.0,
    metric: str = "cosine",
    n_epochs: int = 500,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Dimensionality reduction for episode vectors.
    :param vectors A list of episode vectors
    :param n_components Number of target dimensions
    :param n_neighbors Neighborhood size when learning the manifold structure
    :param min_dist Minimum distance between reduced vectors
    :param metric Distance computation of the input data space
    :param n_epochs Number of epochs
    :param random_state The seed used by the random number generator
    :returns A list of UMAP embeddings
    """
    from umap import UMAP  # Import lazily, since this one takes a while.

    fit = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_epochs=n_epochs,
        random_state=random_state,
    )
    u_emb = fit.fit_transform(vectors.cpu())
    return u_emb
