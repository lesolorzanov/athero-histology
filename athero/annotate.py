"""Annotate (segment) new slides with a trained pipeline.

For every slide in the scratch directory: run the MLP over its features to
get the 64-d embedding, find its k nearest neighbours among the study tiles
in the 64-d space, and transfer their propagated labels. Produces a
per-slide CSV (tile_filename, graph_string, graph_counts) and a rendered
annotation PNG overlaying tiles on the slide.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..constants import CLASS_COLORS
from ..features.datasets import annotatedDF
from ..features.registry import feature_dim
from ..mlp.model import load_pretrained


def knn_annotate(new_embedding, study_embedding, study_labels, k=5):
    """Majority label of the k nearest study tiles for each new tile.

    Returns (labels, counts): the winning label and its vote count (the
    graph_counts column used downstream, 1..k).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(study_embedding)
    _, idx = nn.kneighbors(new_embedding)
    neigh_labels = np.asarray(study_labels)[idx]
    labels, counts = [], []
    for row in neigh_labels:
        values, cts = np.unique(row, return_counts=True)
        winner = values[np.argmax(cts)]
        labels.append(winner)
        counts.append(int(cts.max()))
    return np.asarray(labels), np.asarray(counts)


def render_annotation(csvdf, out_png, pixel_size=224):
    """Plot tile labels as a color-coded segmentation of the slide."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    xs = csvdf["tile_index_x"].values * pixel_size
    ys = -csvdf["tile_index_y"].values * pixel_size
    colors = csvdf["graph_string"].map(CLASS_COLORS).fillna("#aaaaaa")
    ax.scatter(xs, ys, c=colors, s=8, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True,
                    help="scratch dir with <model>_features/ and mask/csv/")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--study", required=True,
                    help="data_for_graphs/<model> with 64D, clustering CSV")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = args.model.lower()

    study_df = pd.read_csv(os.path.join(args.study, "final_clustering.csv"))
    study_emb = np.load(os.path.join(args.study, f"{model}-64D.npy"))
    study_labels = study_df["by_graph"].values

    mlp = load_pretrained(args.weights, in_feats=feature_dim(model),
                          device=device)

    os.makedirs(os.path.join(args.out, f"{model}_csv"), exist_ok=True)
    os.makedirs(os.path.join(args.out, f"{model}_img"), exist_ok=True)

    for feat_file in sorted(
        glob.glob(os.path.join(args.scratch, f"{model}_features", "*.npy"))
    ):
        filename = os.path.basename(feat_file)
        sample_id = filename.split("-" + model)[0]
        features = np.load(feat_file)
        csvdf = pd.read_csv(
            os.path.join(args.scratch, "mask", "csv", sample_id + "_HE.csv")
        )

        dataset = annotatedDF(csvdf, features, "blur")  # labels unused here
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch)
        outs = []
        with torch.inference_mode():
            for batch in loader:
                x = batch["features"].to(device).float()
                outs.append(mlp(x).detach().cpu().numpy())
        embedding = np.vstack(outs)

        labels, counts = knn_annotate(embedding, study_emb, study_labels,
                                     k=args.knn)
        csvdf["graph_string"] = labels
        csvdf["graph_counts"] = counts
        csvdf.to_csv(
            os.path.join(args.out, f"{model}_csv", sample_id + ".csv"),
            index=False,
        )
        render_annotation(
            csvdf, os.path.join(args.out, f"{model}_img", sample_id + ".png")
        )


if __name__ == "__main__":
    main()
