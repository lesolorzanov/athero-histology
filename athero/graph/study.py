"""StudyFeatures: the core object of the graph stage.

Wraps a tiles dataframe, the (64-d) features, a kNN graph, UMAP embedding,
and Leiden partitions. Faithful re-implementation of the original
``studyfeatures.py``, without the legacy MongoDB helpers.
"""

import datetime
import pickle

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from leidenalg import (
    CPMVertexPartition,
    ModularityVertexPartition,
    Optimiser,
    RBConfigurationVertexPartition,
    SignificanceVertexPartition,
    SurpriseVertexPartition,
)
from sklearn.neighbors import kneighbors_graph

PARTITION_TYPES = {
    "CPMVertexPartition": CPMVertexPartition,
    "RBConfigurationVertexPartition": RBConfigurationVertexPartition,
    "ModularityVertexPartition": ModularityVertexPartition,
    "SurpriseVertexPartition": SurpriseVertexPartition,
    "SignificanceVertexPartition": SignificanceVertexPartition,
}


def print_current_time(message):
    print(message, datetime.datetime.now(), flush=True)


class StudyFeatures:
    def __init__(self):
        self.umap_object = None
        self.features = None
        self.tiles_dataframe = None
        self.graph = None
        self.edges = None
        self.partition = None
        self.num_clusters = None

    # ------------------------------------------------------------------ graph
    def create_graph(self, nneighbors=50, metric="minkowski", p=2,
                    n_jobs=-1, inplace=True, initial_edges=None):
        """kNN graph over ``self.features``; returns (igraph.Graph, edges).

        Edges are collected from the kneighbors_graph connectivity matrix;
        optional ``initial_edges`` (e.g. same-class anchors) are prepended.
        """
        print_current_time("Starting graph")
        graph_for_leiden = kneighbors_graph(
            self.features, nneighbors, mode="connectivity",
            metric=metric, p=p, n_jobs=n_jobs,
        )
        edges = list(initial_edges or [])
        print_current_time(">Getting edges by connectivity")
        start = 0
        for i, end in enumerate(graph_for_leiden.indptr[1:]):
            for j in graph_for_leiden.indices[start:end]:
                edges.append([i, int(j)])
            start = end
        edges = np.array(edges)
        print_current_time(">Ended edges")

        g = ig.Graph(len(self.tiles_dataframe), edges)
        if inplace:
            self.edges = edges
            self.graph = g
        return g, edges

    # ----------------------------------------------------------------- leiden
    def do_leiden(self, resolution=2.5,
                  partition_type="RBConfigurationVertexPartition",
                  inplace=True, seed=None):
        """Run Leiden community detection on the current graph."""
        if self.edges is None:
            raise RuntimeError("edges need to be prepared with create_graph")
        print_current_time("Starting Leiden clustering")
        partition = PARTITION_TYPES[partition_type](
            self.graph, resolution_parameter=resolution
        )
        optimiser = Optimiser()
        optimiser.optimise_partition(partition)
        print_current_time("Finished Leiden clustering")
        print(resolution, partition.summary())
        if inplace:
            self.partition = partition
            self.num_clusters = len(partition)
        return partition

    # ------------------------------------------------------------------ umap
    def umap_for_features(self, components=3, umap_kws=None, inplace=True):
        """Fit UMAP on ``self.features`` and store ux/uy/uz columns."""
        if umap_kws is None:
            umap_kws = {
                "n_components": components,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": 42,
            }
        umap_kws.setdefault("n_components", components or 3)
        studyumap = umap.UMAP(**umap_kws)
        embedding = studyumap.fit_transform(self.features)
        if inplace:
            self.umap_object = studyumap
            self.tiles_dataframe["ux"] = embedding[:, 0]
            self.tiles_dataframe["uy"] = embedding[:, 1]
            if umap_kws["n_components"] == 3:
                self.tiles_dataframe["uz"] = embedding[:, 2]
        return studyumap, embedding

    def save(self, directory):
        """Persist edges, UMAP model and the tiles dataframe."""
        if self.edges is not None:
            np.save(f"{directory}/edges.npy", self.edges)
        if self.umap_object is not None:
            with open(f"{directory}/umap_model.pkl", "wb") as f:
                pickle.dump(self.umap_object, f)
        if self.tiles_dataframe is not None:
            self.tiles_dataframe.to_csv(f"{directory}/final_clustering.csv",
                                        index=False)

    def load(self, directory):
        amtiles = len(self.tiles_dataframe)
        self.edges = np.load(f"{directory}/edges.npy")
        self.graph = ig.Graph(amtiles, self.edges)
        with open(f"{directory}/umap_model.pkl", "rb") as f:
            self.umap_object = pickle.load(f)

    # ------------------------------------------------------------------- viz
    def visualize_embedding(self, hue="cluster", palette="rainbow"):
        data = self.tiles_dataframe
        f, axs = plt.subplots(1, 3, figsize=(9, 4), sharey=True, dpi=150)
        sns.scatterplot(data=data, x="ux", y="uy", hue=hue, ax=axs[0],
                         palette=palette, s=2)
        if "uz" in data:
            sns.scatterplot(data=data, x="uy", y="uz", hue=hue, ax=axs[1],
                            palette=palette, s=2)
            sns.scatterplot(data=data, x="ux", y="uz", hue=hue, ax=axs[2],
                            palette=palette, s=2)
        for ax in axs[:-1]:
            ax.get_legend().remove()
        plt.show()
        plt.close()
