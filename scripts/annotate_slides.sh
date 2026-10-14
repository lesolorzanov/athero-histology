#!/usr/bin/env bash
#SBATCH -A <project-id> -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 1-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<user>@<domain>

# Annotate (segment) new slides with a trained pipeline for one model.
# Usage: sbatch annotate_slides.sh <model> <knn>

set -euo pipefail

MODEL=${1:?usage: sbatch annotate_slides.sh <model> <knn>}
KNN=${2:-5}
WORK=${WORK:-/mimer/NOBACKUP/groups/<group>}
REPO=$(cd "$(dirname "$0")/.." && pwd)

module load virtualenv/20.26.2-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a SciPy-bundle/2024.05-gfbf-2024a
source "$WORK/scripts/venv/bin/activate"

cd "$TMPDIR"

# Pipeline state: 64-d study embedding, clustering, MLP weights
cp -r "$WORK/data_for_graphs/$MODEL" ./data_for_graphs
cp "$WORK/model/MLP/MLP-$MODEL-weights.pth" ./

# New slides to annotate: their features and tile CSVs
mkdir -p "${MODEL}_features" mask
cp "$WORK/to_annotate/${MODEL}_features"/*.npy "./${MODEL}_features/"
cp -r "$WORK/mask/csv" ./mask/csv

python "$REPO/athero/annotate.py" \
    --scratch "$TMPDIR" --model "$MODEL" --batch 50 --knn "$KNN" \
    --study "$TMPDIR/data_for_graphs/$MODEL" \
    --weights "./MLP-$MODEL-weights.pth" \
    --out "$WORK/annotations"

echo "Done. Annotations in $WORK/annotations/${MODEL}_csv and _img"
