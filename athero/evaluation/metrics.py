"""Evaluation: confusion matrices (3-level) and per-class AUROC.

Metrics are computed on the held-out validation tiles only; MLP class
probabilities on those tiles are the AUROC inputs. The
confusion-matrix helper mirrors the original ``utils.myOwnCM``: a plain
count matrix, with optional row normalization.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..constants import LEVEL2_CLASSES
from ..viz import myOwnCM, plot_confusion_matrix


def three_level_confusion(evaldf, pred_col="graph_string"):
    """Confusion matrices at Level 1 (LRNC), Level 2 (6 classes) and
    Level 3 (cellular classes, evaluation only)."""
    levels = {
        "level1": ["LRNC"],
        "level2": LEVEL2_CLASSES,
        "level3": ["Neovessels", "MN cells", "Foamcells"],
    }
    out = {}
    for level, classes in levels.items():
        subset = evaldf[evaldf[classes].max(axis=1) > 0]
        y = pd.Categorical(evaldf[classes].idxmax(axis=1), categories=classes).codes
        yhat = pd.Categorical(evaldf[pred_col], categories=classes).codes
        cm = myOwnCM(y, yhat, numclasses=len(classes))
        out[level] = cm
    return out


def per_class_auroc(probs, y_true, classes=LEVEL2_CLASSES):
    """AUROC from MLP class probabilities on held-out validation tiles."""
    aucs = {}
    for i, c in enumerate(classes):
        aucs[c] = roc_auc_score((np.asarray(y_true) == i).astype(int), probs[:, i])
    return aucs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True,
                    help="base dir with annotated_csv/ and GDLannotations/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = args.data
    model = args.model

    gt = pd.concat(
        [
            pd.read_csv(os.path.join(base, "annotated_csv", f))
            for f in os.listdir(os.path.join(base, "annotated_csv"))
            if f.endswith(".csv")
        ]
    )
    pred = pd.concat(
        [
            pd.read_csv(os.path.join(base, "GDLannotations", f"{model}_csv", f))
            for f in os.listdir(os.path.join(base, "GDLannotations", f"{model}_csv"))
            if f.endswith(".csv")
        ]
    )

    merged = gt.merge(
        pred[["tile_filename", "graph_string", "graph_counts"]],
        on="tile_filename",
    )
    cms = three_level_confusion(merged)
    out_dir = args.out or "."
    for level, cm in cms.items():
        plot_confusion_matrix(cm, save=os.path.join(out_dir, f"{model}_{level}.png"))


if __name__ == "__main__":
    main()
