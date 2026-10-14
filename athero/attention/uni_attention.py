"""UNI attention-map explainability.

UNI is a plain (non-hierarchical) ViT-Large/16: each of its 24 blocks has
16 attention heads, giving 384 per-head attention maps per tile. We extract
the head maps with forward hooks, form linear combinations of heads
(weights in {-1, 0, 1}), and search for combinations whose maxima coincide
with cellular structures of interest (multinucleated giant cells,
neovessels) using mask-based scoring: attention inside a manual mask of the
target structure should be maximal, and minimal outside. Chosen combinations
are recorded per structure (Supplementary Table G) and rendered as
attention overlays for expert confirmation.
"""

import numpy as np
import torch


class AttentionExtractor:
    """Per-head attention maps of a timm ViT via forward hooks."""

    def __init__(self, model):
        self.model = model
        self.attention = []  # list over blocks of (B, heads, N, N)
        self.handles = []
        for block in model.blocks:
            self.handles.append(
                block.attn.register_forward_hook(self._hook)
            )

    def _hook(self, module, inputs, output):
        # timm's Attention returns (attn, rolled); keep the full attention
        attn = output[0] if isinstance(output, tuple) else output
        self.attention.append(attn.detach())

    def reset(self):
        self.attention = []

    def __call__(self, x):
        self.reset()
        with torch.inference_mode():
            _ = self.model(x)
        return self.attention  # blocks x (B, H, N, N)

    def close(self):
        for h in self.handles:
            h.remove()


def head_maps(attentions, tile_size=224, patch_size=16):
    """CLS-to-patch attention of every head -> (n_heads, h, w) maps.

    Reshape each head's class-token row into a 14x14 grid and upsample to
    the tile resolution. Returns all 24*16 = 384 maps.
    """
    import torch.nn.functional as F

    grid = tile_size // patch_size
    maps = []
    for block_attn in attentions:          # 24 blocks
        a = block_attn[0]                  # batch element 0: (H, N, N)
        cls_rows = a[:, 0, 1:]             # (H, N-1) patch attentions
        for h in range(cls_rows.shape[0]):
            m = cls_rows[h].reshape(grid, grid)
            m = m - m.min()
            m = m / (m.max() + 1e-8)
            maps.append(
                F.interpolate(
                    torch.as_tensor(m)[None, None].float(),
                    size=(tile_size, tile_size), mode="bilinear",
                )[0, 0].numpy()
            )
    return np.stack(maps)  # (384, tile, tile)


def combine(maps, combination):
    """Linear combination of head maps given {head_index: weight}."""
    out = np.zeros_like(maps[0])
    for idx, weight in combination.items():
        out += weight * maps[idx]
    out = out - out.min()
    return out / (out.max() + 1e-8)


def mask_score(maps, mask, n_candidates=200, rng=None):
    """Search combinations maximising in-mask attention, minimising outside.

    ``mask`` is boolean at tile resolution (True inside the target
    structure). Random {-1,0,1} weight vectors over a few heads are scored as
    mean(in-mask attention) - mean(outside attention); the best combinations
    are returned for visual confirmation by an expert pathologist.
    """
    rng = rng or np.random.default_rng(42)
    inside = mask
    outside = ~mask
    best = []
    for _ in range(n_candidates):
        n_heads = rng.integers(2, 6)
        heads = rng.choice(len(maps), size=n_heads, replace=False)
        weights = rng.choice([-1, 0, 1], size=n_heads)
        combo = dict(zip(heads.tolist(), weights.tolist()))
        m = combine(maps, combo)
        score = m[inside].mean() - m[outside].mean()
        best.append((score, combo))
    best.sort(key=lambda t: -t[0])
    return best
