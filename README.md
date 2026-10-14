# athero-histology

Weakly supervised segmentation of atherosclerotic plaque histology from H&E
whole-slide images, using foundation-model features, a weakly supervised MLP
projection, and graph clustering with label propagation.

This repository contains the analysis pipeline for the BiKE (Biobank of
Karolinska Endarterectomies) plaque histology study. It was restructured from
the original research scripts; the logic is unchanged, only the organization.
The pipeline is environment-independent: every stage is a plain Python script
that runs on a laptop (GPU if available, CPU otherwise) or any cluster.

## Pipeline overview

```
WSIs (H&E, Olympus VS200, 20x, DZI pyramids)
   |  1. Tissue masking (Otsu) and tile listing (224 px tiles, 56 px overlap)
   v
Feature extraction (per foundation model)
   |  2. Five pre-trained feature extractors, benchmarked:
   |     Virchow2 (2560-d), UNI (1024-d), UNI2 (1536-d),
   |     TissueConcepts HViT (512-d), Ciga ResNet18 (512-d)
   v
Weakly supervised MLP projection
   |  3. 4-layer-block MLP trained on weak ROI annotations maps features to a
   |     64-d embedding (multi-class focal loss, AdaBelief)
   v
Graph construction and clustering
   |  4. kNN graph (k=50) on the 64-d embedding; Leiden community detection
   v
Label propagation
   |  5. Community-majority assignment spreads the 6 tissue-class labels
   |     (IPH, CH, IF, CA, RM, CT) to all tiles, annotated or not
   v
Evaluation and analyses
      6. Confusion matrices and AUROC on the held-out validation tiles,
         interobserver comparison, classification confidence (mode of the
         100 nearest neighbours), attention-map explainability (UNI)
```

## Repository layout

```
extract_features.py   Run one foundation model over all slides -> features .npy
train_mlp.py           Train the weakly supervised MLP for one model
build_graph.py         64-d projection, kNN graph, Leiden, label propagation
annotate_slides.py     Segment new slides (kNN label transfer)
evaluate_metrics.py    Confusion matrices (3 levels) + AUROC on validation tiles
make_partition.py      Slide-level stratified train/validation split

athero/                The importable package behind the entry points
  constants.py           Class hierarchy, canonical ordering, colors
  dzi.py                 DZI pyramid geometry helpers
  masking.py             Otsu tissue detection and tile listing
  features/              Model registry, datasets, extraction
  mlp/                   FeatToClass MLP, training on weak annotations
  graph/                 kNN graph, Leiden clustering, label propagation
  annotate.py            Segmentation of new slides
  evaluation/            Confusion matrices, AUROC, classification confidence
  attention/             UNI per-head attention maps and combination search
  viz.py                 Plotting helpers (confusion matrices, UMAP, PLY export)
configs/
  models.yaml           Feature-extractor registry (weights, dims, transforms)
  cluster_maps.json     Per-model Leiden community -> tissue class assignments
partition/               Slide-level train/validation partitioning
```

Each entry-point script is a thin wrapper: the logic lives in the `athero`
package, the script just parses command-line arguments and calls it.

## Label hierarchy

Three annotation levels are used:

| Level | Classes | Use |
|-------|---------|-----|
| 1 (macro) | LRNC | evaluation only |
| 2 (tissue) | IPH, CH, IF, CA, RM, CT | training (the 6 classes) |
| 3 (cellular) | NV, MNC, FC | evaluation only (attention maps) |

Annotations are ROI-based and multi-label: a tile can carry several labels at
once. For MLP training a single class is derived per tile (the highest-index
set label, see `athero/mlp/train.py`). Only the six Level-2 tissue classes
are used for training.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Use `--device cpu` on machines without a GPU (feature extraction will be
slow; the MLP and graph stages are fine).

## Usage

Run from the repository root. All paths are your own; nothing is
cluster-specific.

```bash
# 1. List tissue tiles per slide
python -m athero.masking --dzis <dir of DZI pyramids> --out <csv out dir>

# 2. Extract features for one model (uni | uni2 | v2 | ciga | tc)
python extract_features.py --model v2 \
    --tiles-dir <dir with <slide>_files/ folders> \
    --csv-dir <dir with <slide>_HE.csv> \
    --weights <model weights dir> --out <features dir>

# 3. Partition the 50 annotated slides into train/validation (slide-level,
#    stratified by class prevalence)
python make_partition.py --annotations <dir with <slide>_HE.csv> \
    --out data_partition.csv

# 4. Train the weakly supervised MLP for one model
python train_mlp.py --model v2 --data <data_for_MLP/v2>

# 5. Project features to 64-d, build the kNN graph, run Leiden,
#    and propagate labels by community majority
python build_graph.py --model v2 --data <data_for_graphs/v2> \
    --weights <MLP-v2-weights.pth> \
    --annotations <common_maindf_weak_annot.csv>

# 6. Segment (annotate) new slides
python annotate_slides.py --model v2 --scratch <dir with v2_features/*.npy> \
    --batch 50 --knn 5 --study <data_for_graphs/v2> \
    --weights <MLP-v2-weights.pth> --out <annotations dir>

# 7. Evaluation: confusion matrices and AUROC
python evaluate_metrics.py --model v2 --data <work dir>
```

Feature extraction is run once per model (five times in total); every stage
after it is parameterised by the same model name, and outputs are written to
per-model directories.

## Data

The BiKE whole-slide images and annotations are not part of this repository
and are available on reasonable request under a data transfer agreement. The
pipeline expects:

- DZI pyramids of the WSIs (one folder per slide, `*_files/` tile grids),
- one CSV of tile metadata + weak annotations per slide
  (`tile_filename`, tissue proportions, blur, one 0/1 column per class),
- feature files `<slide>-<model>-features.npy` produced by step 2.

## Citation / contact

Code accompanying the manuscript "Foundation models comparison for
identification of patterns specific to plaque morphology in cardiovascular
histology". Please see the manuscript for details of the methods.
