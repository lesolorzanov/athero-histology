"""Feature extraction from WSI tiles with the benchmarked foundation models.

For each slide tar of tiles in the scratch directory, this module runs the
selected extractor over all tissue tiles and saves one
``<slide>-<model>-features.npy`` array, plus the slide's tile dataframe for
downstream steps.

Model-specific details (feature dimension and aggregation):

- v2  (Virchow2): concatenate CLS token with the mean of the 5 region tokens
                  -> 2560-d (fp16 inputs, fp32 outputs).
- uni (UNI): timm forward_features, CLS token -> 1024-d.
- uni2 (UNI2): timm forward_features, mean of patch + register tokens
                  -> 1536-d.
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


def extract_batch(model_name, model, images):
    """Model-specific embedding of a batch of preprocessed tile images."""
    with torch.inference_mode():
        if model_name == "v2":
            output = model(images)  # (B, 261, 1280)
            class_token = output[:, 0, :]              # (B, 1280)
            patch_tokens = output[:, 1:, :]            # (B, 260, 1280)
            region_tokens = patch_tokens.reshape(-1, 5, 52, 1280).mean(2)
            return torch.cat([class_token, region_tokens.mean(1)], dim=-1)
        if model_name == "tc":
            feature_pyramid = model["encoder"](images.to(model.device))
            return model["squeeezer"](feature_pyramid[-1])[1]  # hidden vector
        if model_name in ("uni", "uni2"):
            return model(images)
        # ciga and other CNNs: penultimate activations already exposed
        return model(images)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True, help="scratch dir with tars of tiles")
    ap.add_argument("--model", required=True, choices=registry.MODELS)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--weights", default=None, help="model weights directory")
    ap.add_argument("--out", default=None, help="features output dir")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = registry.load_model(args.model, args.weights, device)
    transform = build_transform(args.model, model)

    out_dir = args.out or os.path.join(args.scratch, f"{args.model}_features")
    os.makedirs(out_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(args.scratch, "tar", "*")))
    for slide_dir in file_list:
        slide_id = os.path.basename(slide_dir).split("_HE_files_level")[0]
        out_path = os.path.join(out_dir, f"{slide_id}-{args.model}.npy")
        if os.path.exists(out_path):
            continue
        csv_file = os.path.join(args.scratch, "csv", f"{slide_id}_HE.csv")
        dataset = wsiHEDataset(slide_dir, csv_file, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

        outputs = []
        for images in loader:
            images = images.to(device)
            if args.model == "v2":
                images = images.half()
            emb = extract_batch(args.model, model, images)
            outputs.append(emb.detach().cpu().numpy())

        outputs = np.concatenate(outputs)
        np.save(out_path, outputs)
        print(f"done extracting features for {slide_id}: {outputs.shape}", flush=True)


if __name__ == "__main__":
    main()
