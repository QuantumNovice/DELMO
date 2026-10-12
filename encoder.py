"""M: the frozen CLIP ViT-B/32 image encoder (OpenAI weights) and the exact input preprocessing.

Views arrive as uint8 grey images (224 x 224, see :mod:`delmo.render`), rasters as float [0, 1] RGB
(see :func:`delmo.topology.raster_image`). Both are expanded to RGB, resized to 224 if needed and normalised
with CLIP's mean and standard deviation before ``encode_image``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIZE = 224


def to_rgb(x: torch.Tensor, size: int = SIZE, mean=CLIP_MEAN, std=CLIP_STD) -> torch.Tensor:
    """uint8 grey (N,H,W) or float [0,1] (N,3,H,W) -> normalised (N,3,size,size)."""
    if x.dtype == torch.uint8:
        x = (x.float() / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    m = torch.tensor(mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    s = torch.tensor(std, device=x.device, dtype=x.dtype)[None, :, None, None]
    return (x - m) / s


class ClipEncoder:
    """Frozen CLIP image tower. ``weights=None`` builds an untrained network (tests, offline)."""

    def __init__(self, device="cpu", weights: str | None = "openai", arch: str = "ViT-B-32"):
        import open_clip

        self.device = torch.device(device)
        self.model = open_clip.create_model(arch, pretrained=weights).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.visual.output_dim

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocessed (N,3,224,224) -> (N,512) float32."""
        amp = torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda")
        with amp:
            return self.model.encode_image(images.to(self.device)).float()

    @torch.no_grad()
    def phi(self, views: torch.Tensor, raster: torch.Tensor) -> torch.Tensor:
        """views uint8 (B,4,224,224) + raster float (B,3,224,224) -> phi (B,5,512): front, top, side, iso, topology."""
        b = views.shape[0]
        ims = to_rgb(views.flatten(0, 1).to(self.device))
        g = to_rgb(raster.to(self.device))
        e = self(torch.cat([ims, g]))
        return torch.cat([e[:4 * b].view(b, 4, -1), e[4 * b:].view(b, 1, -1)], 1)
