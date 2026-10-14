"""Feature extraction from WSI tiles with the benchmarked foundation models.

Location-independent: give it a directory of slide tile folders, a directory
of per-slide tile CSVs, and an output directory. Runs on a laptop GPU, a
cluster, or CPU (slow). One .npy of features per slide is written:

    <out>/<slide>-<model>-features.npy

Model-specific details (feature dimension and aggregation):

- v2  (Virchow2): concatenate CLS token with the mean of the 5 region tokens
                  -> 2560-d (fp16 inputs, fp32 outputs).
- uni (UNI): timm forward, CLS token -> 1024-d.
- uni2 (UNI2): timm forward, mean of patch + register tokens -> 1536-d.
- tc  (TissueConcepts): HViT squeeezer output, hidden vector -> 512-d.
- ciga (tenpercent ResNet18): penultimate 512-d activations.
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from . import registry
from .datasets import wsiHEDataset


def build_transform(model_name, model=None):
    """Standard 224x224 ImageNet-normalised transform (256 for TissueConcepts)."""
    if model_name == "v2":
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    size = 256 if model_name == "tc" else 224
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def extract_features(model_name, model, dataset, device, batch=50):
    """Run one extractor over a wsiHEDataset -> (n_tiles, feature_dim) array."""
    loader = DataLoader(dataset, batch_size=batch, shuffle=False)
    outputs = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device)
            if model_name == "v2":
                images = images.half()
            if model_name == "v2":
                output = model(images)  # (B, 261, 1280)
                class_token = output[:, 0, :]
                patch_tokens = output[:, 1:, :]
                region_tokens = patch_tokens.reshape(-1, 5, 52, 1280).mean(2)
                emb = torch.cat([class_token, region_tokens.mean(1)], dim=-1)
            elif model_name == "tc":
                feature_pyramid = model["encoder"](images.to(model.device))
                emb = model["squeeezer"](feature_pyramid[-1])[1]
            else:  # uni, uni2, ciga
                emb = model(images)
            outputs.append(emb.detach().cpu().float().numpy())
    return np.concatenate(outputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=registry.MODELS,
                    help="feature extractor to run")
    ap.add_argument("--tiles-dir", required=True,
                    help="directory with one tile folder per slide "
                         "(e.g. BIKE_1007_files/ with the tile JPEGs)")
    ap.add_argument("--csv-dir", required=True,
                    help="directory with one <slide>_HE.csv per slide "
                         "(produced by athero.masking)")
    ap.add_argument("--weights", default=None,
                    help="model weights directory; omit to fetch from the hub "
                         "where supported (v2, tc)")
    ap.add_argument("--out", required=True, help="output directory for .npy files")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default=None,
                    help="cuda / cuda:0 / cpu (default: best available)")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = registry.load_model(args.model, args.weights, device)
    transform = build_transform(args.model, model)

    os.makedirs(args.out, exist_ok=True)

    slide_dirs = sorted(
        d for d in glob.glob(os.path.join(args.tiles_dir, "*"))
        if os.path.isdir(d) and d.endswith("_files")
    )
    if not slide_dirs:
        raise SystemExit(f"no <slide>_files directories found in {args.tiles_dir}")

    for slide_dir in slide_dirs:
        slide_id = os.path.basename(slide_dir).replace("_files", "")
        out_path = os.path.join(args.out, f"{slide_id}-{args.model}-features.npy")
        if os.path.exists(out_path):
            print(f"skipping {slide_id} (features exist)", flush=True)
            continue
        csv_file = os.path.join(args.csv_dir, f"{slide_id}_HE.csv")
        dataset = wsiHEDataset(slide_dir, csv_file, transform=transform)
        feats = extract_features(args.model, model, dataset, device,
                                 batch=args.batch)
        np.save(out_path, feats)
        print(f"done {slide_id}: {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
