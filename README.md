# athero-histology

Weakly supervised segmentation of atherosclerotic plaque histology from H&E
whole-slide images, using foundation-model features, a weakly supervised MLP
projection, and graph clustering with label propagation.

This repository contains the analysis pipeline for the BiKE (Biobank of
Karolinska Endarterectomies) plaque histology study. It was restructured from
the original research scripts; the logic is unchanged, only the organization.

## Pipeline overview

```
WSIs (H&E, Olympus VS200, 20x, DZI pyramids)
   │  1. Tissue masking (Otsu) and tile listing (224 px tiles, 56 px overlap)
   ▼
Feature extraction (per foundation model)
   │  2. Five pre-trained feature extractors, benchmarked:
   │     Virchow2 (2560-d), UNI (1024-d), UNI2 (1536-d),
   │     TissueConcepts HViT (512-d), Ciga ResNet18 (512-d)
   ▼
Weakly supervised MLP projection
   │  3. 4-layer-block MLP trained on weak ROI annotations maps features to a
   │     64-d embedding (multi-class focal loss, AdaBelief)
   ▼
Graph construction and clustering
   │  4. kNN graph (k=50) on the 64-d embedding; Leiden community detection
   ▼
Label propagation
   │  5. Community-majority assignment spreads the 6 tissue-class labels
   │     (IPH, CH, IF, CA, RM, CT) to all tiles, annotated or not
   ▼
Evaluation and analyses
      6. Confusion matrices and AUROC on the held-out validation tiles,
         interobserver comparison, classification confidence (mode of the
         100 nearest neighbours), attention-map explainability (UNI)
```

## Repository layout

```
athero/
  constants.py        Class hierarchy, canonical ordering, colors
  dzi.py              Deep Zoom Image (DZI) pyramid geometry helpers
  masking.py          Otsu tissue detection and tile tissue-content listing
  features/           Model registry, datasets, feature extraction
  mlp/                FeatToClass MLP, training on weak annotations
  graph/              kNN graph, Leiden clustering, label propagation
  annotate.py         Annotate (segment) new slides via the 64-d embedding
  evaluation/         Confusion matrices, AUROC, classification confidence
  attention/          UNI per-head attention maps and combination search
  viz.py              Plotting helpers (confusion matrices, UMAP, PLY export)
configs/
  models.yaml         Feature-extractor registry (weights, dims, transforms)
  cluster_maps.json   Per-model Leiden community -> tissue class assignments
scripts/              SLURM job scripts (feature extraction, annotation)
partition/            Slide-level train/validation partitioning
notebooks/           Mapping of the original exploratory notebooks
```

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

GPU jobs are expected to run under SLURM (see `scripts/`). All heavy steps
write their outputs to a working directory (`$WORK` below), never inside the
repository.

## Usage

```bash
# 1. List tissue tiles per slide
python -m athero.masking --dzis $WORK/dzis --out $WORK/mask/csv

# 2. Extract features for one model (uni | uni2 | v2 | ciga | tc)
python -m athero.features.extract --model v2 --scratch $TMPDIR --batch 50

# 3. Partition the 50 annotated slides into train/validation (slide-level,
#    stratified by class prevalence)
python -m partition.split --annotations $WORK/annotated_csv --out $WORK/data_partition.csv

# 4. Train the weakly supervised MLP for one model
python -m athero.mlp.train --model v2 --data $WORK/data_for_MLP/v2

# 5. Project features to 64-d, build the kNN graph, run Leiden,
#    and propagate labels by community majority
python -m athero.graph.run --model v2 --data $WORK/data_for_graphs/v2 \
    --weights $WORK/model/MLP/MLP-v2-weights.pth \
    --annotations $WORK/data_for_graphs/common_maindf_weak_annot.csv

# 6. Segment (annotate) all remaining slides
python -m athero.annotate --model v2 --scratch $TMPDIR --batch 50 --knn 5 \
    --study $WORK/data_for_graphs/v2 --weights $WORK/model/MLP/MLP-v2-weights.pth \
    --out $WORK/annotations

# 7. Evaluation: confusion matrices, AUROC, confidence maps
python -m athero.evaluation.metrics --model v2 --data $WORK
python -m athero.evaluation.confidence --model v2 --data $WORK
```

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
