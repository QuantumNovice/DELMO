"""DELMO network: G (feature translation), source aggregation and C (classifier) on frozen CLIP vectors.

    X_i  = [four rendered views, adjacency raster of the depth-3 neighbourhood]
    phi  = M(X_i)     frozen CLIP ViT-B/32, one 512-d vector per view and one for the raster
    xi   = G(phi)     LayerNorm -> Linear(512, 512) -> GELU -> dropout -> Linear(512, 512) -> LayerNorm,
                      applied to every source vector
    z    = attention pooling over the five sources (learned embedding per source) -> projection
    y~   = C(z)       LayerNorm -> dropout -> Linear(512, n_classes)

The module and parameter names match the released checkpoint (``checkpoint/delmo_clip_vitb32.pt``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn

SOURCES = ("front", "top", "side", "iso", "topology")


class Translation(nn.Module):
    """G: native encoder space -> shared space (``mlp`` with depth 2 in the released model)."""

    def __init__(self, in_dim: int, hidden: int = 512, kind: str = "mlp", dropout: float = 0.1, depth: int = 2):
        super().__init__()
        if kind == "identity":
            self.net = nn.Identity()
            return
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden)]
        if kind == "mlp":
            for _ in range(max(0, depth - 1)):
                layers += [nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)]
        elif kind != "linear":
            raise ValueError(f"unknown translation {kind!r}")
        layers.append(nn.LayerNorm(hidden))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SourceAggregator(nn.Module):
    """Attention pooling of the source vectors; ``last_weights`` shows how much each view / the topology counts."""

    def __init__(self, hidden: int, sources: Sequence[str]):
        super().__init__()
        self.idx = [SOURCES.index(s) for s in sources]
        self.embed = nn.Parameter(torch.zeros(len(SOURCES), hidden))
        nn.init.trunc_normal_(self.embed, std=0.02)
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.last_weights: torch.Tensor | None = None

    def forward(self, x):  # (B, S, H)
        x = x + self.embed[self.idx].to(x.dtype)
        w = self.score(x).squeeze(-1).softmax(1)
        self.last_weights = w.detach()
        return (x * w.unsqueeze(-1)).sum(1)


class Fusion(nn.Module):
    """Projection of the pooled vector (with one encoder there is nothing to gate)."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, pooled: Mapping[str, torch.Tensor]):
        return self.proj(next(iter(pooled.values())))


class Classifier(nn.Module):
    def __init__(self, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class DELMO(nn.Module):
    def __init__(self, n_classes: int, sources: Sequence[str] = SOURCES, in_dim: int = 512, hidden: int = 512,
                 translation: str = "mlp", dropout: float = 0.1, encoder: str = "clip", depth: int = 2):
        super().__init__()
        self.encoder, self.sources = encoder, list(sources)
        self.G = nn.ModuleDict({encoder: Translation(in_dim, hidden, translation, dropout, depth)})
        self.agg = nn.ModuleDict({encoder: SourceAggregator(hidden, sources)})
        self.fusion = Fusion(hidden, dropout)
        self.C = Classifier(hidden, n_classes, dropout)

    def forward(self, phi: torch.Tensor, return_all: bool = False):
        """``phi``: (B, S, 512) CLIP vectors in ``self.sources`` order."""
        e = self.encoder
        xi = self.G[e](phi.float())
        pooled = {e: self.agg[e](xi)}
        z = self.fusion(pooled)
        logits = self.C(z)
        if not return_all:
            return logits
        return {"logits": logits, "xi": xi, "z": z, "source_weights": self.agg[e].last_weights}

