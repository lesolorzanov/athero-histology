# Notebooks

The exploratory and figure notebooks were kept out of the package and are
re-run from the installed modules. Original notebook -> new location mapping:

| Original notebook | Where it lives now |
|---|---|
| `featureextraction.py` + `featureextraction.sh` | `extract_features.py` + `athero/features/` |
| `10-train_MLP_from_models.ipynb` | `train_mlp.py` + `athero/mlp/train.py` |
| `11-CreateGraphs-*.ipynb` (one per model) | `build_graph.py` + `athero/graph/" (model is a flag) |
| `12-AnnotateNewImage.ipynb`, `annotate.py` + `annotate.sh` | `annotate_slides.py` + `athero/annotate.py` |
| `13-AnalysisofResults.ipynb`, `createCMfigures.ipynb`, `createAUCROC.ipynb` | `evaluate_metrics.py` + `athero/evaluation/metrics.py` |
| `15-countConfidence.ipynb` | `athero/evaluation/confidence.py` |
| `14-GetRepresentationsOfClusters.ipynb`, `graphto3D.ipynb` | `athero/viz.py` (`write_ply`) |
| `UNI_attention*.ipynb`, `16-FindMNCellsForViTAttention.ipynb` | `athero/attention/uni_attention.py` |
| `partition_test.py` + `partition_params.sh` | `make_partition.py` + `partition/split.py` |
| `mask.py`, `mask_dzi.ipynb` | `athero/masking.py` |
| `DZICalculator.py`, `dzi_together.py`, `dzi_puzzletogether.ipynb` | `athero/dzi.py` |
| `studyfeatures.py` | `athero/graph/study.py` |
| `utils.py`, `vsmlplib.py` | `athero/viz.py`, `athero/mlp/model.py`, `athero/features/datasets.py` |
| `studyfeatures.py` legacy Mongo helpers | dropped (unused in the final analysis) |
