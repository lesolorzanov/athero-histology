"""Plotting and export helpers.

Confusion matrices (count matrix helper ``myOwnCM``), UMAP scatter plots and
PLY point-cloud export of the 3-d UMAP embedding for visual inspection of
clusters in 3-d viewers.
"""

import numpy as np


# ----------------------------------------------------------------- confusion
def norm_cm(cm, eps=1e-6):
    return cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + eps)


def myOwnCM(y, yhat, numclasses=6, numeric=True, labelsdict=None, dtype=None):
    """Count confusion matrix without sklearn: cm[y, yhat]."""
    cm = np.zeros((numclasses, numclasses),
                  dtype=dtype if dtype is not None else np.int64)
    for t, p in zip(np.asarray(y).ravel(), np.asarray(yhat).ravel()):
        if numeric:
            cm[int(t), int(p)] += 1
        else:
            cm[labelsdict[t], labelsdict[p]] += 1
    return cm


def plot_confusion_matrix(cm, classes=None, norm_cm=None, cmap="viridis",
                          save=None, pad_inches=None, vmin=0.0, vmax=1.0,
                          ax=None, annot_kws=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if norm_cm is None:
        norm_cm = norm_cm(cm)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sns.heatmap(norm_cm, annot=cm, fmt="d", cmap=cmap, ax=ax,
                vmin=vmin, vmax=vmax, annot_kws=annot_kws or {})
    if classes is not None:
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Annotated")
    if save and fig is not None:
        fig.savefig(save, bbox_inches="tight", pad_inches=pad_inches or 0.1)
    return ax


# --------------------------------------------------------------------- umap
def plot_umap(indf, colx="ux", coly="uy", color_col=None,
              s=5, frac=1.0, save=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = indf.sample(frac=frac)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    sns.scatterplot(
        data=data, x=colx, y=coly, hue=color_col, s=s,
        linewidth=0, ax=ax,
    )
    if color_col and data[color_col].nunique() > 30:
        ax.get_legend().remove()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------- ply
def write_ply(name, adf, colorcol=None, umapcols=("ux", "uy", "uz")):
    """Export the 3-d UMAP embedding as a colored PLY point cloud.

    ``colorcol`` must hold hex color strings, one per tile.
    """
    with open(name, "w") as f:
        n = len(adf)
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for _, row in adf.iterrows():
            x, y, z = (row[c] for c in umapcols)
            rgb = (128, 128, 128)
            if colorcol is not None and isinstance(row[colorcol], str):
                h = row[colorcol].lstrip("#")
                rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
            f.write(f"{x} {y} {z} {rgb[0]} {rgb[1]} {rgb[2]}\n")
