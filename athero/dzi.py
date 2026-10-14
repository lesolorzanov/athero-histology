"""Deep Zoom Image (DZI) pyramid geometry helpers.

Whole-slide images are stored as DZI pyramids (one ``.dzi`` XML descriptor
plus a ``<slide>_files/<level>/`` grid of JPEG tiles). These helpers read the
pyramid geometry and assemble full tiles at any level.

Based on the DZI specification as described in the "inside deep zoom" blog
series (gasi.ch).
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from skimage import io


def get_maximum_level(w, h):
    return int(math.ceil(math.log(max(w, h), 2)))


def compute_levels(w, h, tilesize, debug=False):
    """Return (columns, rows, widths, heights) for every pyramid level."""
    widths, heights, columns, rows = [], [], [], []
    maxlevel = get_maximum_level(w, h)
    for _ in range(int(maxlevel), -1, -1):
        c = math.ceil(w / tilesize)
        r = math.ceil(h / tilesize)
        columns.insert(0, int(c))
        rows.insert(0, int(r))
        widths.insert(0, int(w))
        heights.insert(0, int(h))
        w = math.ceil(w / 2)
        h = math.ceil(h / 2)
        if debug:
            print(f"level {len(columns)-1} is {widths[0]} x {heights[0]}")
    return columns, rows, widths, heights


def get_tile_position(col, row, tilesize=1024, overlap=1):
    """Pixel offset of a tile in the level image (overlap default 1)."""
    offset_x = 0 if col == 0 else overlap
    offset_y = 0 if row == 0 else overlap
    return (col * tilesize) - offset_x, (row * tilesize) - offset_y


def get_whts_from_dzi(dzifile):
    """Parse a .dzi descriptor -> (width, height, tilesize, overlap)."""
    if not os.path.isfile(dzifile):
        raise FileNotFoundError(f"DZI does not exist: {dzifile}")
    tree = ET.parse(dzifile)
    root = tree.getroot()
    width = height = tile_size = overlap = None
    for child in root.iter():
        if any(size in child.tag.lower() for size in ["size"]):
            width = int(child.attrib["Width"])
            height = int(child.attrib["Height"])
        elif "tile" in child.tag.lower():
            tile_size = int(child.attrib["TileSize"])
            overlap = int(child.attrib["Overlap"])
    return width, height, tile_size, overlap


def tile_path(dzi_dir, level, col, row, extension="jpeg"):
    """Path of one pyramid tile inside a ``<slide>_files`` directory."""
    slide = os.path.basename(os.path.normpath(dzi_dir)).replace("_files", "")
    return os.path.join(
        dzi_dir, str(level), f"{slide}_{col}_{row}.{extension}"
    )


def assemble_level(dzi_dir, level, expected_shape, tile_size=1024, overlap=1):
    """Assemble all tiles of one level into a single image array."""
    cols, rows = expected_shape
    canvas = None
    for c in range(cols):
        for r in range(rows):
            path = tile_path(dzi_dir, level, c, r)
            if not os.path.exists(path):
                continue
            x, y = get_tile_position(c, r, tile_size, overlap)
            img = io.imread(path)
            if canvas is None:
                canvas = np.zeros(
                    (rows * tile_size, cols * tile_size, img.shape[2]),
                    dtype=img.dtype,
                )
            h, w = img.shape[:2]
            canvas[y : y + h, x : x + w] = img
    return canvas
