"""Project-wide constants: label hierarchy, class ordering, colors."""

# Canonical Level-2 tissue classes (the six training classes), canonical order.
# Order matters: it defines the bit used for each class in the multi-label
# -> single-label encoding (cnum), and the ordering of confusion-matrix rows.
LEVEL2_CLASSES = [
    "Connective tissue",
    "Remnant media",
    "IPH",
    "Chol",
    "Inflammation",
    "calcification",
]

# Short aliases used across tables and figures.
CLASS_ALIASES = {
    "Connective tissue": "CT",
    "Remnant media": "RM",
    "IPH": "IPH",
    "Chol": "CH",
    "Inflammation": "IF",
    "calcification": "CA",
}

# Level 1 (macro) and Level 3 (cellular) classes: evaluation only.
LEVEL1_CLASSES = ["LRNC"]
LEVEL3_CLASSES = ["Neovessels", "MN cells", "Foamcells"]

# All annotation columns in a weak-annotation CSV.
ALL_ANNOTATION_COLUMNS = LEVEL1_CLASSES + LEVEL2_CLASSES + LEVEL3_CLASSES

CLASS_TO_INDEX = {c: i for i, c in enumerate(LEVEL2_CLASSES)}
INDEX_TO_CLASS = {i: c for i, c in enumerate(LEVEL2_CLASSES)}

# Figure colors (also used for PLY exports of the 3-d UMAP embedding).
CLASS_COLORS = {
    "calcification": "#518c2b",
    "IPH": "#e84387",
    "LRNC": "#4ac5d6",
    "Chol": "#c893f2",
    "Foam cells": "#046467",
    "Neovessels": "#ffd500",
    "Connective tissue": "#fb879b",
    "Inflammation": "#5e003c",
    "Remnant media": "#ee6a13",
    "Foamcells": "#046467",
    "MN cells": "#c36e3e",
    "undefined": "#aaaaaa",
    "unannotated": "#aaaaaa",
}

# Fixed hyperparameters of the pipeline (see manuscript Methods).
GRAPH_KNN = 50          # k of the kNN graph on the 64-d embedding
CONFIDENCE_KNN = 100     # neighbourhood for classification confidence
LEIDEN_PARTITION = "RBConfigurationVertexPartition"
LEIDEN_RESOLUTION = 2.5
MLP_EMBEDDING_DIM = 64
MLP_EPOCHS = 35
MLP_BATCH_SIZE = 500
MLP_LR = 1e-6
FOCAL_GAMMA = 3
OVERSAMPLE_PER_CLASS = 15000
