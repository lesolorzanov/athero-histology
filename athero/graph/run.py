"""End-to-end graph stage for one model.

Projects extracted features to the 64-d embedding with the trained MLP,
builds the kNN graph, runs UMAP and Leiden, assigns community-majority
labels, and saves everything needed by later stages:

    <data>/<model>-64D.npy
    <data>/edges.npy
    <data>/umap_model.pkl
    <data>/final_clustering.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..constants import GRAPH_KNN, LEIDEN_PARTITION, LEIDEN_RESOLUTION
from ..features.datasets import annotatedDF
from ..features.registry import feature_dim
from ..mlp.model import load_pretrained
from .propagate import community_majority_labels, propagate_labels
from .study import StudyFeatures


def project_to_64d(mlp, df, features, device, batch_size=50000):
    """Apply the MLP (head removed) to all tiles -> 64-d array."""
    dataset = annotatedDF(df, features, "cnum2")
    loader = DataLoader(dataset, batch_size=batch_size)
    outs = []
    mlp.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["features"].to(device).float()
            outs.append(mlp(x).detach().cpu().numpy())
    return np.vstack(outs)


def run(model_name, data_dir, weights_path, annotations_csv, device="cpu",
        knn=GRAPH_KNN, resolution=LEIDEN_RESOLUTION,
        partition_type=LEIDEN_PARTITION):
    df = pd.read_csv(os.path.join(data_dir, "data_frame.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    # join weak annotations (one row per tile, 0/1 columns per class)
    annot = pd.read_csv(annotations_csv)
    akeep = [c for c in annot.columns if c != "tile_filename"]
    df = pd.concat([df, annot[akeep]], axis=1)

    features = np.load(os.path.join(data_dir, f"{model_name}-features.npy"))

    mlp = load_pretrained(weights_path, in_feats=feature_dim(model_name),
                          device=device)
    outs = project_to_64d(mlp, df, features, device)
    np.save(os.path.join(data_dir, f"{model_name}-64D.npy"), outs)

    study = StudyFeatures()
    study.features = outs
    study.tiles_dataframe = df

    study.create_graph(nneighbors=knn)
    partition = study.do_leiden(resolution=resolution,
                                partition_type=partition_type)
    study.tiles_dataframe["cluster"] = partition.membership

    study.umap_for_features(components=3)

    clust_to_max = community_majority_labels(study.tiles_dataframe)
    study.tiles_dataframe = propagate_labels(study.tiles_dataframe, clust_to_max)

    study.save(data_dir)
    return study


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="data_for_graphs/<model>")
    ap.add_argument("--weights", required=True, help="MLP-<model>-weights.pth")
    ap.add_argument("--annotations", required=True,
                    help="common_maindf_weak_annot.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args.model, args.data, args.weights, args.annotations, device=device)


if __name__ == "__main__":
    main()
