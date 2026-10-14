"""Tissue detection on DZI tiles.

Each slide is processed at a fixed pyramid level: tiles are turned to
grayscale, gray-white background is filtered, Otsu's method separates tissue
from background, and per-tile tissue proportion and blur are recorded. The
output CSV per slide (``<slide>_HE.csv``) lists every tile with its tissue
metadata; only tiles with tissue content proceed to feature extraction.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from skimage import io


def grayscale(img):
    """ITU-R BT.601 luma."""
    return np.dot(img[..., 0:3], [0.299, 0.587, 0.114])


def invert_grayscale(gray_img):
    return abs(gray_img - 255)


def gray_filter(img, tolerance=10):
    """Set near-gray (background pen) pixels to white."""
    r = np.asarray(img[..., 0], dtype="int64")
    g = np.asarray(img[..., 1], dtype="int64")
    b = np.asarray(img[..., 2], dtype="int64")
    is_gray = (
        (np.abs(r - g) <= tolerance)
        & (np.abs(g - b) <= tolerance)
        & (np.abs(r - b) <= tolerance)
    )
    out = img.copy()
    for c in range(3):
        out[..., c][is_gray] = 255
    return out


def otsu_interclass_variance(img, thr):
    return np.nansum(
        [
            np.mean(group) * np.var(img, where=group)
            for group in [img >= thr, img < thr]
        ]
    )


def otsu_thresholding(img):
    """Otsu threshold on an uint8 grayscale tile; None if degenerate."""
    candidate_range = range(int(np.min(img)) + 1, int(np.max(img)))
    if len(candidate_range) > 1:
        return min(candidate_range, key=lambda th: otsu_interclass_variance(img, th))
    return None


def tiles_with_tissue(name, level=13, override_maxlevel=None, compute_blur=False,
                      skip_borders=True):
    """Summarise tissue content of every tile of one slide.

    Returns a DataFrame with one row per tile: tile indices, tissue pixel
    count and proportion, and (optionally) a blur estimate.
    """
    dzi_dir = f"{name}_files"
    if override_maxlevel is None:
        dzi_xml = name + ".dzi"
        if not os.path.exists(dzi_xml):
            levels = sorted(int(l) for l in os.listdir(dzi_dir) if l.isdigit())
            level = levels[-1] if levels else level
    tiles = sorted(glob.glob(os.path.join(dzi_dir, str(level), "*.jpg")))
    records = []
    for tpath in tiles:
        img = io.imread(tpath)
        gray = grayscale(img)
        thr = otsu_thresholding(gray.astype(np.uint8))
        if thr is None:
            continue
        tissue = gray >= thr  # dark pixels = tissue
        records.append(
            {
                "tile_filename": os.path.relpath(tpath, os.path.dirname(dzi_dir)),
                "tissue_pixels": int(tissue.sum()),
                "tissue_proportion": float(tissue.mean()),
                "blur": compute_blur,
            }
        )
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dzis", required=True, help="directory of slide DZI folders")
    ap.add_argument("--out", required=True, help="output directory for per-slide CSVs")
    ap.add_argument("--level", type=int, default=13, help="pyramid level to analyse")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for dzi in sorted(glob.glob(os.path.join(args.dzis, "*"))):
        slide = os.path.basename(dzi)
        csv_path = os.path.join(args.out, f"{slide}_HE.csv")
        if os.path.exists(csv_path):
            continue
        df = tiles_with_tissue(os.path.join(args.dzis, slide), level=args.level)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
