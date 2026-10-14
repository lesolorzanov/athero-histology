# SLURM job scripts

Cluster jobs for the ALVIS/NAISS infrastructure. Adapt the account id,
partition, mail user, and `$WORK` paths at the top of each script.

| Script | Stage | GPU |
|--------|-------|-----|
| `extract_features.sh <model>` | feature extraction from slide tars | yes |
| `annotate_slides.sh <model> <knn>` | segment new slides with a trained pipeline | yes |

CPU-heavy steps (partition testing, graph construction, evaluation) run in
notebooks or directly with `python -m athero...`; see the repository README.
