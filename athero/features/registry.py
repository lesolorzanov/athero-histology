"""Feature-extraction model registry for the five benchmarked extractors.

Each entry records how to build the model, the expected input size and the
dimensionality of the extracted embedding, matching the manuscript:

| key | model                       | type | feature dim |
|-----|-----------------------------|------|-------------|
| v2  | Virchow2 (Paige AI)        | ViT  | 2560        |
| uni | UNI (MahmoodLab)           | ViT  | 1024        |
| uni2| UNI2 (MahmoodLab)          | ViT  | 1536        |
| tc  | TissueConcepts HViT        | HViT | 512         |
| ciga| Ciga tenpercent ResNet18   | CNN  | 512         |

Feature dimensions are concatenations of the class token and pooled region
tokens where applicable (e.g. Virchow2: CLS + mean of 5 region tokens).
"""

import os

import torch


FEATURE_DIMS = {
    "v2": 2560,
    "uni": 1024,
    "uni2": 1536,
    "tc": 512,
    "ciga": 512,
}

MODELS = sorted(FEATURE_DIMS)


def feature_dim(model_name):
    try:
        return FEATURE_DIMS[model_name]
    except KeyError:
        raise ValueError(f"Not a valid model: {model_name}. "
                         f"Valid options: {MODELS}")


def _load_uni(weights_dir, device):
    """UNI: plain ViT-Large/16, timm implementation, local weights."""
    import timm

    local_bin = os.path.join(weights_dir, "pytorch_model.bin")
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(local_bin, map_location="cpu"), strict=True)
    return model


def _load_uni2(weights_dir, device):
    """UNI2: ViT-Giant/14, timm implementation, local weights."""
    import timm
    import timm.layers

    kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embedded_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **kwargs)
    model.load_state_dict(
        torch.load(os.path.join(weights_dir, "uni2_pytorch_model.bin"),
                   map_location="cpu"),
        strict=True,
    )
    return model


def _load_virchow2(weights_dir, device):
    """Virchow2: HuggingFace model, CLS + 5 mean-pooled region tokens."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("paige-ai/Virchow2", torch_dtype=torch.float16)
    return model


def _load_ciga(weights_dir, device):
    """Ciga: ResNet18 self-supervised on 10% ImageNet ('tenpercent')."""
    import torchvision

    model = torchvision.models.__dict__["resnet18"](pretrained=False)
    state = torch.load(os.path.join(weights_dir, "tenpercent_resnet18.ckpt"),
                       map_location="cpu")
    state_dict = state["state_dict"]
    # checkpoint keys are prefixed with "model."; drop the classifier head.
    state_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items()
        if not k.startswith("model.fc")
    }
    model.load_state_dict(state_dict)
    model.fc = torch.nn.Sequential()  # penultimate 512-d output
    return model


def _load_tissueconcepts(weights_dir, device):
    """TissueConcepts HViT: hierarchical ViT via the mmm package."""
    from mmm.api.M3Model import M3Model, M3_MODELS, UNICORN_ENCODER

    return M3Model(M3_MODELS[UNICORN_ENCODER])


LOADERS = {
    "uni": _load_uni,
    "uni2": _load_uni2,
    "v2": _load_virchow2,
    "ciga": _load_ciga,
    "tc": _load_tissueconcepts,
}


def load_model(model_name, weights_dir, device):
    """Instantiate a feature extractor on ``device`` and set eval mode."""
    model = LOADERS[model_name](weights_dir, device)
    model = model.to(device)
    model.eval()
    return model
