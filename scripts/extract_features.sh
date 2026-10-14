#!/usr/bin/env bash
#SBATCH -A <project-id> -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 2-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<user>@<domain>

# Extract features from all slides with one foundation model.
# Usage: sbatch extract_features.sh <model>  (uni | uni2 | v2 | ciga | tc)

set -euo pipefail

MODEL=${1:?usage: sbatch extract_features.sh <model>}
WORK=${WORK:-/mimer/NOBACKUP/groups/<group>}
REPO=$(cd "$(dirname "$0")/.." && pwd)

module load virtualenv/20.26.2-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a SciPy-bundle/2024.05-gfbf-2024a
source "$WORK/scripts/venv/bin/activate"

cd "$TMPDIR"
echo "in $TMPDIR"

# Copy model weights and tile CSVs to node-local scratch
case "$MODEL" in
  uni)  cp "$WORK/model/uni/pytorch_model.bin" ./ ;;
  uni2) cp "$WORK/model/uni2/uni2_pytorch_model.bin" ./ ;;
  ciga) cp "$WORK/model/ciga/tenpercent_resnet18.ckpt" ./ ;;
esac
cp -r "$WORK/mask/csv" ./csv

# Transfer slide tars
mkdir -p tar
for filename in "$WORK"/dzistar/*.tar; do
    name=$(basename "${filename%.*}")
    if ls "$WORK/${MODEL}_features/"*"${name%%_*}"*.npy >/dev/null 2>&1; then
        echo "npy exists for ${name}, skipping"
        continue
    fi
    mkdir "tar/$name"
    cp "$filename" "./${name}.tar"
    tar -xf "${name}.tar" -C "tar/$name/"
    rm "${name}.tar"
done

python "$REPO/athero/features/extract.py" \
    --scratch "$TMPDIR" --model "$MODEL" --batch 50 \
    --weights "$WORK/model/$MODEL" --out "$WORK/${MODEL}_features"

echo "Done. Moved outputs to $WORK/${MODEL}_features"
