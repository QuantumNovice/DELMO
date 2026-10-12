"""X1: the four rendered views of an IFC entity, exactly as the training corpus was rendered.

IfcOpenShell geometry in world coordinates -> trimesh (``process=False``) -> pyrender: transparent
background, ambient light 0.2, grey metallic-roughness material (base 0.5, metallic 0.1, roughness 0.7,
smooth shading), three directional lights of intensity 0.5, perspective camera ``yfov = pi/3``, 800 x 800
RGBA. Camera distance is twice the bounding-box diagonal; front at (d cos 45°, d sin 45°, 0.7 d), top at
(0, 0, 1.5 d), side at (1.5 d, 0, 0.3 d), iso at 0.577 x 1.2 d on every axis, all looking at the centroid.
Each image is cropped to the object and padded by 20 px, then composited on white, converted to grey,
letterboxed to a square and resized to 224 px (:func:`to_view`).

On a machine without a display set ``PYOPENGL_PLATFORM=egl`` (Linux) before running.
"""

from __future__ import annotations

import numpy as np

VIEWS = ("front", "top", "side", "iso")


def _orient(cam: np.ndarray, center: np.ndarray) -> None:
    d = center - cam[:3, 3]
    d = d / np.linalg.norm(d)
    right = np.cross(d, np.array([0, 0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, d)
    cam[:3, 0], cam[:3, 1], cam[:3, 2] = right, up, -d


def camera_poses(center: np.ndarray, distance: float) -> list[tuple[np.ndarray, str]]:
    poses = []
    a = np.pi / 4
    cam = np.eye(4)
    cam[:3, 3] = center + np.array([distance * np.cos(a), distance * np.sin(a), distance * 0.7])
    _orient(cam, center)
    poses.append((cam.copy(), "front"))
    cam = np.eye(4)
    cam[:3, 3] = center + np.array([0, 0, distance * 1.5])
    cam[:3, 0], cam[:3, 1], cam[:3, 2] = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    poses.append((cam.copy(), "top"))
    cam = np.eye(4)
    cam[:3, 3] = center + np.array([distance * 1.5, 0, distance * 0.3])
    _orient(cam, center)
    poses.append((cam.copy(), "side"))
    cam = np.eye(4)
    cam[:3, 3] = center + np.full(3, distance * 1.2 * 0.577)
    _orient(cam, center)
    poses.append((cam.copy(), "iso"))
    return poses


def render_mesh(verts: np.ndarray, faces: np.ndarray, renderer) -> dict:
    """Four cropped, padded RGBA PIL images of one mesh."""
    import pyrender
    import trimesh
    from PIL import Image

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.is_empty:
        return {}
    distance = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) * 2.0
    out = {}
    for pose, name in camera_poses(mesh.centroid, distance):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.2, 0.2, 0.2, 1.0])
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.5, 0.5, 0.5, 1.0], metallicFactor=0.1,
                                                 roughnessFactor=0.7)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True))
        for t in ([0, 0, 0], [1, 1, 1], [-1, -1, 1]):
            lp = np.eye(4)
            lp[:3, 3] = t
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.5), pose=lp)
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        img = Image.fromarray(color, mode="RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            padded = Image.new("RGBA", (img.width + 40, img.height + 40), (0, 0, 0, 0))
            padded.paste(img, (20, 20))
            img = padded
        out[name] = img
    return out


def geometry_settings():
    import ifcopenshell.geom

    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, True)
    return s


def render_product(product, settings, renderer) -> dict:
    """The four views of one IfcProduct ({} when it has no body geometry)."""
    import ifcopenshell.geom

    try:
        sh = ifcopenshell.geom.create_shape(settings, product)
        v = np.array(sh.geometry.verts).reshape(-1, 3)
        f = np.array(sh.geometry.faces).reshape(-1, 3)
    except Exception:  # noqa: BLE001 - no shape, no views (as in the dataset scripts)
        return {}
    if len(v) == 0 or len(f) == 0:
        return {}
    return render_mesh(v, f, renderer)


def to_view(im, size: int = 224) -> np.ndarray:
    """PIL image -> uint8 (size, size): RGBA on white, grey, letterboxed square, bilinear resize."""
    from PIL import Image

    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("L")
    w, h = im.size
    s = max(w, h)
    sq = Image.new("L", (s, s), 255)
    sq.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(sq.resize((size, size), Image.BILINEAR), np.uint8)


def load_view(path, size: int = 224) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return to_view(im, size)
