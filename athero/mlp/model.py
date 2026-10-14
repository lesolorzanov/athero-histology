"""Weakly supervised MLP that projects foundation-model features to 64-d.

Architecture (``FeatToClass``): four linear blocks with batch norm + ReLU,
shrinking the extractor-specific input dimension (2560/1536/1024/512) through
350-256-128 down to a 64-d penultimate embedding, followed by a 6-way
classifier head. The 64-d penultimate layer is the representation used for
graph construction; the classifier head provides the weakly supervised signal.
"""

import torch


class FeatToClass(torch.nn.Module):
    def __init__(self, in_feats=512, num_classes=6, embedding_dim=64):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=in_feats, out_features=350, bias=True)
        self.lin1 = torch.nn.Linear(in_features=350, out_features=256, bias=True)
        self.bn = torch.nn.BatchNorm1d(256)
        self.act = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(in_features=256, out_features=256, bias=True)
        self.lin3 = torch.nn.Linear(in_features=256, out_features=128, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.act1 = torch.nn.ReLU()
        self.lin4 = torch.nn.Linear(in_features=128, out_features=128, bias=True)
        self.lin5 = torch.nn.Linear(in_features=128, out_features=embedding_dim, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.act2 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features=embedding_dim, out_features=num_classes, bias=True)

    def embed(self, x):
        """64-d embedding (penultimate activations)."""
        x = self.lin(x)
        x = self.lin1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        return self.fc(self.embed(x))


def load_pretrained(path, in_feats, num_classes=6, embedding_dim=64, device="cpu"):
    """Load trained MLP weights, with the classifier head removed.

    The returned module maps features to the 64-d embedding, equivalent to
    the original trick ``mymlp.fc = torch.nn.Sequential()``.
    """
    model = FeatToClass(in_feats=in_feats, num_classes=num_classes,
                        embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    model.fc = torch.nn.Sequential()
    model.to(device)
    model.eval()
    return model
