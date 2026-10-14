"""Datasets for feature extraction and MLP training."""

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class wsiHEDataset(Dataset):
    """BiKE whole-slide H&E tile dataset.

    Iterates over the tiles of one slide, listed in a per-slide tissue CSV
    (produced by :mod:`athero.masking`), reading each tile JPEG from its DZI
    pyramid directory.
    """

    def __init__(self, dzi_path, csv_file, transform=None):
        """
        Arguments:
            dzi_path (str): path to the slide's ``<slide>_files`` directory.
            csv_file (str): per-slide tile CSV (tile_filename, ...).
            transform (callable, optional): preprocessing transform.
        """
        self.dzi_path = dzi_path
        self.csv_file = csv_file
        self.tiles = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row = self.tiles.iloc[idx]
        imgname = row["tile_filename"].split("/")[-1]
        img = Image.open(f"{self.dzi_path}/{imgname}.jpeg")
        if self.transform:
            img = self.transform(img)
        return img


class annotatedDF(Dataset):
    """Tiles with weak annotations joined to their extracted features.

    ``class_col`` names the column of ``annot_df`` holding the (integer)
    training label; features are indexed positionally against the dataframe.
    """

    def __init__(self, annot_df, features, class_col):
        self.annot_df = annot_df
        self.features = np.asarray(features)
        self.class_col = class_col

    def __len__(self):
        return len(self.annot_df)

    def __getitem__(self, idx):
        row = self.annot_df.iloc[idx]
        return {
            "features": self.features[idx, ...],
            "class": row[self.class_col],
        }
