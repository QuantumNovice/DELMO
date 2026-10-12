"""Classify IFC entities with the released model and route each one (accept / auto-correct / refer).

Routing, with p the temperature-calibrated class probabilities and y the declared IFC class:

    predicted == y                         -> accept
    predicted != y and max(p) >= tau       -> auto   (corrected to the predicted class)
    predicted != y and max(p) <  tau       -> refer  (human verification)

tau = 0.95 by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint"


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def route(declared: str, predicted: str, confidence: float, tau: float) -> str:
    if declared == predicted:
        return "accept"
    return "auto" if confidence >= tau else "refer"


class Predictor:
    """Frozen CLIP (M) + the released G / C weights.

    ``weights=None`` skips the CLIP download (an untrained encoder, for tests only).
    """

    def __init__(self, checkpoint: str | Path = CHECKPOINT, device: str | None = None, weights: str | None = "openai"):
        import torch

        from .encoder import ClipEncoder
        from .model import DELMO
        from .topology import Rasterizer

        ck = Path(checkpoint)
        self.config = json.loads((ck / "config.json").read_text(encoding="utf-8"))
        self.classes = json.loads((ck / "classes.json").read_text(encoding="utf-8"))
        vocab = json.loads((ck / "topology_vocab.json").read_text(encoding="utf-8"))
        a = self.config["architecture"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DELMO(len(self.classes), a["sources"], a["in_dim"], a["hidden"], a["translation"], a["dropout"],
                           depth=a.get("translation_depth", 2))
        sd = torch.load(ck / self.config["weights"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()
        self.encoder = ClipEncoder(self.device, weights)
        self.rasterizer = Rasterizer(vocab, a["max_nodes"])
        self.temperature = float(self.config["calibration"]["temperature"])
        self.tau = float(self.config["calibration"]["tau"])

    # ------------------------------------------------------------------ core
    def predict_arrays(self, views: np.ndarray, rasters: np.ndarray) -> np.ndarray:
        """views uint8 (B,4,224,224) in front/top/side/iso order, rasters uint8 (B,3,128,128) -> calibrated (B,C)."""
        import torch

        from .topology import raster_image

        v = torch.from_numpy(np.ascontiguousarray(views))
        g = raster_image(torch.from_numpy(np.ascontiguousarray(rasters)), 224, self.config["architecture"]["raster"])
        phi = self.encoder.phi(v, g)
        with torch.no_grad():
            p = self.model(phi.to(self.device)).float().softmax(-1).cpu().numpy()
        return apply_temperature(p, self.temperature)

    def _rows(self, probs: np.ndarray, meta: list[dict], tau: float) -> list[dict]:
        out = []
        for m, q in zip(meta, probs):
            k = int(q.argmax())
            row = {**m, "predicted": self.classes[k], "confidence": float(q[k]),
                   "top3": "; ".join(f"{self.classes[i]} {q[i]:.3f}" for i in np.argsort(-q)[:3])}
            row["decision"] = route(m.get("declared"), row["predicted"], row["confidence"], tau)
            row["declared_in_training_classes"] = m.get("declared") in self.classes
            out.append(row)
        return out

    # ------------------------------------------------------------------ IFC file
    def predict_ifc(self, ifc_path: str | Path, tau: float | None = None, batch: int = 16, limit: int | None = None):
        """Render every IfcProduct, build its neighbourhood from the same file, classify and route.
        Returns a pandas DataFrame (one row per entity with a body)."""
        import ifcopenshell
        import pandas as pd
        import pyrender

        from .render import VIEWS, geometry_settings, render_product, to_view
        from .topology import ifc_graph, neighbourhoods

        tau = self.tau if tau is None else tau
        f = ifcopenshell.open(str(ifc_path))
        prods = [p for p in f.by_type("IfcProduct") if p.Representation is not None]
        cls, gid, edges = ifc_graph(f)
        nb = neighbourhoods(cls, gid, edges, [p.id() for p in prods])
        settings = geometry_settings()
        renderer = pyrender.OffscreenRenderer(800, 800)
        rows, bv, bg, bm = [], [], [], []

        def flush():
            if bm:
                rows.extend(self._rows(self.predict_arrays(np.stack(bv), np.stack(bg)), bm, tau))
                bv.clear(), bg.clear(), bm.clear()

        try:
            for p in prods:
                if limit and len(rows) + len(bm) >= limit:
                    break
                imgs = render_product(p, settings, renderer)
                if len(imgs) < 4:
                    continue
                n = nb[p.id()]
                bv.append(np.stack([to_view(imgs[v]) for v in VIEWS]))
                bg.append(self.rasterizer(n["classes"], n["src"], n["dst"], n["kind"]))
                bm.append({"global_id": p.GlobalId, "step_id": p.id(), "name": p.Name, "declared": p.is_a()})
                if len(bm) >= batch:
                    flush()
            flush()
        finally:
            renderer.delete()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ rendered views
    def predict_renders(self, folder: str | Path, ifc_path: str | Path | None = None, tau: float | None = None,
                        batch: int = 32):
        """Classify ``<GlobalId>-<IfcClass>-<view>.png`` renders (the dataset layout). With ``ifc_path`` the
        neighbourhoods come from that file; otherwise each raster holds the entity alone."""
        import pandas as pd

        from .render import VIEWS, load_view

        tau = self.tau if tau is None else tau
        groups: dict[tuple[str, str], dict[str, Path]] = {}
        for p in sorted(Path(folder).glob("*.png")):
            gid, sep, rest = p.stem.partition("-Ifc")
            cls, sep2, view = ("Ifc" + rest).rpartition("-")
            if sep and sep2 and view in VIEWS:
                groups.setdefault((gid, cls), {})[view] = p
        nbs = {}
        if ifc_path:
            import ifcopenshell

            from .topology import ifc_graph, neighbourhoods

            f = ifcopenshell.open(str(ifc_path))
            cls, gidmap, edges = ifc_graph(f)
            by_gid = {g: i for i, g in gidmap.items()}
            want = [by_gid[g] for g, _ in groups if g in by_gid]
            nbs = {gidmap[k]: v for k, v in neighbourhoods(cls, gidmap, edges, want).items()}
        rows, bv, bg, bm = [], [], [], []
        for (gid, cls), vs in groups.items():
            fill = next(vs[v] for v in ("iso", "front", "side", "top") if v in vs)
            bv.append(np.stack([load_view(vs.get(v, fill)) for v in VIEWS]))
            n = nbs.get(gid, {"classes": [cls], "src": [], "dst": [], "kind": []})
            bg.append(self.rasterizer(n["classes"], n["src"], n["dst"], n["kind"]))
            bm.append({"global_id": gid, "declared": cls, "has_topology": gid in nbs})
            if len(bm) >= batch:
                rows.extend(self._rows(self.predict_arrays(np.stack(bv), np.stack(bg)), bm, tau))
                bv, bg, bm = [], [], []
        if bm:
            rows.extend(self._rows(self.predict_arrays(np.stack(bv), np.stack(bg)), bm, tau))
        return pd.DataFrame(rows)


def summary(df, tau: float) -> dict:
    """Decision counts and, over entities whose declared class is a training class, agreement with it."""
    out = {"tau": tau, "entities": int(len(df))}
    if not len(df):
        return out
    out["decisions"] = {k: int(v) for k, v in df.decision.value_counts().items()}
    known = df[df.declared_in_training_classes]
    if len(known):
        out["agreement_with_declared"] = float((known.declared == known.predicted).mean())
    return out


def write_corrected(ifc_path: str | Path, df, out_path: str | Path) -> list[dict]:
    """Corrected copy of the IFC with every ``auto`` decision applied (relationships kept)."""
    from .reclass import reclass_file

    changes: Iterable = [(r.global_id, r.predicted) for r in df[df.decision == "auto"].itertuples()]
    return reclass_file(ifc_path, out_path, changes, changelog=Path(out_path).with_suffix(".changes.csv"))
