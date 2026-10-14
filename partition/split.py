"""Slide-level train/validation partition of the 50 annotated slides.

The partition is slide-level: all tiles from a slide belong to the same
partition, so no slide (and no pixel) is shared between training and
validation. Slides are assigned greedily so that both partitions preserve
the class prevalence of the six Level-2 tissue classes as closely as
possible (stratified by class prevalence, *not* balanced).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from athero.constants import LEVEL2_CLASSES


def slide_class_prevalence(csv_dir):
    """Per-slide fraction of annotated tiles carrying each class."""
    rows = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        df = pd.read_csv(f)
        n = len(df)
        row = {"name": os.path.basename(f).replace("_HE.csv", "")}
        for c in LEVEL2_CLASSES:
            row[c] = df[c].sum() / max(n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def partition_slides(prevalence, train_fraction=0.6, seed=42):
    """Greedy stratified assignment of slides to train/validation."""
    rng = np.random.default_rng(seed)
    slides = list(prevalence["name"])
    rng.shuffle(slides)
    n_train = int(round(train_fraction * len(slides)))
    target = prevalence[LEVEL2_CLASSES].mean()

    train, val = [], []
    train_sum = np.zeros(len(LEVEL2_CLASSES))
    val_sum = np.zeros(len(LEVEL2_CLASSES))
    p = prevalence.set_index("name")
    for s in slides:
        v = p.loc[s, LEVEL2_CLASSES].values.astype(float)
        # assign to the partition whose prevalence would move furthest
        # from the cohort target if it took this slide
        cost_train = np.abs((train_sum + v) / (len(train) + 1) - target).sum()
        cost_val = np.abs((val_sum + v) / (len(val) + 1) - target).sum()
        remaining_train = n_train - len(train)
        remaining_val = (len(slides) - n_train) - len(val)
        if remaining_train <= 0:
            take_train = False
        elif remaining_val <= 0:
            take_train = True
        else:
            take_train = cost_train <= cost_val
        (train if take_train else val).append(s)
        if take_train:
            train_sum += v
        else:
            val_sum += v

    out = pd.DataFrame(
        {
            "name": slides,
            "t": ["train" if s in train else "test" for s in slides],
        }
    )
    out["annotated"] = out["name"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True,
                    help="directory of <slide>_HE.csv weak annotation files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prevalence = slide_class_prevalence(args.annotations)
    partition = partition_slides(prevalence,
                                 train_fraction=args.train_fraction,
                                 seed=args.seed)
    partition.to_csv(args.out, index=False)
    print(partition.groupby("t").count())


if __name__ == "__main__":
    main()
