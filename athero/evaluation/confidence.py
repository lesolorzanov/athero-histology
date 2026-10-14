"""Classification confidence of propagated labels.

Confidence of a tile's label is the frequency of that label among the
tile's 100 nearest neighbours in the 64-d embedding (max 100/100;
approximately one sixth minimum by pigeonhole). This is the definition used
for the per-model confidence maps and the replicability analysis across
consecutive sections.
"""

import numpy as np
import pandas as pd

from ..constants import CONFIDENCE_KNN


def confidence(embedding, labels, k=CONFIDENCE_KNN):
    """Vote frequency of each tile's own label among its k nearest neighbours."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(embedding)
    _, idx = nn.kneighbors(embedding)
    neigh_labels = np.asarray(labels)[idx]
    own = np.asarray(labels)[:, None]
    return (neigh_labels == own).sum(axis=1) / k


def confidence_per_class(df, label_col="graph_string",
                          count_col="graph_counts", classes=None):
    """Distribution of confidence (vote counts) per class, for Fig. 5."""
    if classes is None:
        from ..constants import LEVEL2_CLASSES
        classes = LEVEL2_CLASSES
    out = {}
    for c in classes:
        subset = df[df[label_col] == c]
        counts = subset[count_col].value_counts().sort_index()
        out[c] = counts / counts.max() if len(counts) else counts
    return out
