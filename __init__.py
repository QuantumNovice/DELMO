"""DELMO: multi-view + topology classification of IFC entities with confidence-gated class correction.

phi = M(X) frozen CLIP ViT-B/32 on four rendered views and the depth-3 adjacency raster,
xi = G(phi) two-layer MLP feature translation, y~ = C(xi) linear classifier; a disagreement with the declared class
is corrected automatically at calibrated confidence >= tau (0.95) and referred to a person below it.
"""

__version__ = "1.0.0"

from .predictor import Predictor  # noqa: E402,F401
