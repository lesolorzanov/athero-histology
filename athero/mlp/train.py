"""Training of the weakly supervised MLP on the sparse ROI annotations.

Steps per model:

1. Load the annotated tiles dataframe and their features
   (``data_for_MLP/<model>/``).
2. Resolve the multi-label weak annotations to a single training class per
   tile (``cnum`` bit encoding, highest set bit wins).
3. Oversample each class to a fixed number of tiles (with replacement) to
   counter the heavy class imbalance of ROI annotation.
4. Train with multi-class focal loss (gamma=3) and the AdaBelief optimizer.
5. Report confusion matrices on train and validation tiles; save weights.

The slide-level train/validation partition is respected throughout: all tiles
from a slide belong to exactly one partition, and validation tiles are never
used for training.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from adabelief_pytorch import AdaBelief
from focal_loss.focal_loss import FocalLoss
from torch.utils.data import DataLoader

from ..constants import (
    CLASS_TO_INDEX,
    FOCAL_GAMMA,
    LEVEL2_CLASSES,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_LR,
    OVERSAMPLE_PER_CLASS,
)
from ..features.datasets import annotatedDF
from ..viz import plot_confusion_matrix, myOwnCM
from .model import FeatToClass


def binlist(n):
    """Bits of ``n``, least significant first."""
    bits = []
    if n == 0:
        return [0]
    while n > 0:
        bits.append(n & 1)
        n = n >> 1
    return bits


def resolve_single_class(df, class_columns=LEVEL2_CLASSES):
    """Multi-label -> single training class.

    Each tile's labels are encoded as a bitmask ``cnum`` over the canonical
    class order; tiles with several labels are assigned the highest-index
    set label (the most specific tissue class in our ordering).
    """
    df = df.copy()
    df["cnum"] = 0
    df["cstring"] = ""
    for i, c in enumerate(class_columns):
        df["cnum"] += df[c] * 2**i
        df["cnum"] = df["cnum"].astype(int)
        df["cstring"] += df.apply(
            lambda row: c + ";" if row[c] == 1 else "", axis=1
        )
    df["cnum2"] = df["cnum"].copy().astype(int)
    df["cstring2"] = ""
    for i, row in df.iterrows():
        cnum = int(row["cnum"])
        if cnum == 0:
            continue
        bits = binlist(cnum)
        highest = max(j for j, v in enumerate(bits) if v == 1)
        df.loc[i, "cstring2"] = class_columns[highest]
        df.loc[i, "cnum2"] = 2**highest
    df["class"] = df["cstring2"]
    return df


def make_dataloader(df, features, class_col="class_num", shuffle=True):
    data = annotatedDF(df, features, class_col)
    return DataLoader(data, batch_size=MLP_BATCH_SIZE, shuffle=shuffle)


def predict(model, loader, device):
    model.eval()
    outs = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["features"].to(device).float()
            outs.append(model(x).detach().cpu().numpy())
    return np.vstack(outs)


def train(model, traindl, device, epochs=MLP_EPOCHS, lr=MLP_LR,
          num_classes=6, focal_alpha=None):
    if focal_alpha is None:
        focal_alpha = [0.7] + [1.0] * (num_classes - 1)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=focal_alpha,
                          task_type="multi-class", num_classes=num_classes)
    optimizer = AdaBelief(model.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.999),
                          weight_decay=1e-4, weight_decouple=True, rectify=False)

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in traindl:
            optimizer.zero_grad()
            x, y = batch["features"].to(device).float(), batch["class"].to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
        print(f"Epoch {epoch} | Loss {np.mean(epoch_losses):6.2f}", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="feature model key (v2, uni, ...)")
    ap.add_argument("--data", required=True,
                    help="data_for_MLP/<model> with data_frame.csv and features .npy")
    ap.add_argument("--out", default=None, help="weights output path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model.lower()

    df = pd.read_csv(os.path.join(args.data, "data_frame.csv"))
    features = np.load(os.path.join(args.data, f"{model_name}-features.npy"))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df["orig_index"] = df.index.values

    df = resolve_single_class(df)

    traindf = df[df["part"] == "train"].copy()
    # oversample each class to a fixed size (with replacement)
    traindf = pd.concat(
        [
            traindf[traindf["cstring2"] == c].sample(
                OVERSAMPLE_PER_CLASS, replace=True
            )
            for c in LEVEL2_CLASSES
        ]
    )
    testdf = df[df["part"] == "test"].copy()

    traindf["class_num"] = traindf["class"].map(CLASS_TO_INDEX)
    testdf["class_num"] = testdf["class"].map(CLASS_TO_INDEX)

    trainnpy = features[traindf["orig_index"].values]
    testnpy = features[testdf.index.values]

    traindl = make_dataloader(traindf, trainnpy, shuffle=True)
    testdl = make_dataloader(testdf, testnpy, shuffle=False)

    model = FeatToClass(in_feats=trainnpy.shape[1], num_classes=len(LEVEL2_CLASSES))
    model.to(device)
    train(model, traindl, device)

    outs = predict(model, testdl, device)
    res = np.argmax(outs, axis=1)
    testdf["numres"] = res
    cm = myOwnCM(testdf["class_num"], testdf["numres"], numclasses=len(LEVEL2_CLASSES))
    plot_confusion_matrix(cm, classes=LEVEL2_CLASSES, cmap="nipy_spectral")

    out = args.out or f"MLP-{model_name}-weights.pth"
    torch.save(model.state_dict(), out)
    print(f"saved weights to {out}")


if __name__ == "__main__":
    main()
