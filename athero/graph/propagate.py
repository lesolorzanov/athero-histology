"""Community-majority label propagation.

After Leiden clustering, each community is assigned the majority tissue
class of its annotated tiles; this label is then propagated to *all* tiles of
the community (annotated or not), which is how the pipeline segments whole
slides from sparse ROI annotations. Ties/empty communities never occurred in
practice; a tie would be broken deterministically by class order.
"""

import json
import os

import pandas as pd


def community_majority_labels(tiles_dataframe, cluster_col="cluster",
                              label_col="cstring2"):
    """Map each community to the majority annotated class of its tiles.

    Returns a dict {community_id: class_name}.
    """
    annotated = tiles_dataframe[tiles_dataframe[label_col].notna()]
    percluster = (
        annotated[[cluster_col, label_col, "tile_filename"]]
        .groupby(by=[cluster_col, label_col])
        .count()
        .reset_index()
        .sort_values(by="tile_filename", ascending=False)
    )
    clust_to_max = {}
    for u, incluster in percluster.groupby(cluster_col):
        clust_to_max[u] = incluster.iloc[0][label_col]
    return clust_to_max


def propagate_labels(tiles_dataframe, clust_to_max, cluster_col="cluster",
                     out_col="by_graph"):
    """Assign every tile the class of its community."""
    tiles_dataframe[out_col] = tiles_dataframe[cluster_col].map(clust_to_max)
    return tiles_dataframe


def load_cluster_map(model_name, config_dir):
    """Load the fixed community->class map recorded for a model.

    After inspecting each model's clustering, the community assignments were
    frozen (see ``configs/cluster_maps.json``) so that downstream annotation
    of new slides is deterministic.
    """
    with open(os.path.join(config_dir, "cluster_maps.json")) as f:
        return {int(k): v for k, v in json.load(f)[model_name].items()}
