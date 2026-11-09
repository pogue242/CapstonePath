# shared code for the photo scripts
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EARTH = 6378137.0
ORIGIN = math.pi * EARTH
TILE_PX = 256


LAYER = os.environ.get("PHOTO_LAYER", "lb")
RUN = os.environ.get("PHOTO_RUN", "")
SQUARES_PLAN = os.environ.get("PHOTO_SQUARES_PLAN", "tiles_squares.json")
MODEL_RUN = os.environ.get("PHOTO_MODEL_RUN", RUN)


def layer_key(layer: str = LAYER) -> str:
    return {"lb": "wien_ogd_2022", "basemap": "basemap_ortho"}.get(layer, f"wien_ogd_{layer}")


def photo_dir(city: str) -> Path:
    return ROOT / "data" / "interim" / city / "rebuild" / "photo"


def run_dir(city: str) -> Path:
    p = photo_dir(city) / "runs" / RUN if RUN else photo_dir(city)
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_path(city: str) -> Path:
    city = os.environ.get("PHOTO_MODEL_CITY", city)
    d = photo_dir(city) / "runs" / MODEL_RUN if MODEL_RUN else photo_dir(city)
    return d / "seg_final.pt"


def tile_path(city: str, z: int, r: int, c: int, layer: str = None) -> Path:
    return ROOT / "data" / "raw" / "imagery" / city / layer_key(layer or LAYER) / str(z) / str(r) / f"{c}.jpeg"


def tile_m(z: int) -> float:
    return 2 * ORIGIN / 2 ** z


def mosaic(city: str, z: int, tiles: list) -> tuple[np.ndarray | None, tuple]:
    from PIL import Image
    rows = [t[0] for t in tiles]
    cols = [t[1] for t in tiles]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    img = np.zeros(((r1 - r0 + 1) * TILE_PX, (c1 - c0 + 1) * TILE_PX, 3), dtype=np.uint8)
    missing = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            p = tile_path(city, z, r, c)
            if not p.exists():
                missing += 1
                continue
            with Image.open(io.BytesIO(p.read_bytes())) as im:
                img[(r - r0) * TILE_PX:(r - r0 + 1) * TILE_PX, (c - c0) * TILE_PX:(c - c0 + 1) * TILE_PX] = \
                    np.asarray(im.convert("RGB"))
    s = tile_m(z)
    bounds = (-ORIGIN + c0 * s, ORIGIN - (r1 + 1) * s, -ORIGIN + (c1 + 1) * s, ORIGIN - r0 * s)
    if missing > 0.05 * (r1 - r0 + 1) * (c1 - c0 + 1):
        return None, bounds
    return img, bounds


def transform_for(bounds, shape):
    from rasterio.transform import from_bounds
    return from_bounds(*bounds, shape[1], shape[0])


def rasterize_mask(mask_gdf, bounds, shape) -> np.ndarray:
    from rasterio.features import rasterize
    tr = transform_for(bounds, shape)
    out = np.zeros(shape, dtype=np.uint8)
    for kind, val in (("pos", 1), ("ignore", 255)):
        g = mask_gdf.loc[mask_gdf["kind"] == kind].geometry
        g = [x for x in g if x is not None and not x.is_empty]
        if g:
            m = rasterize([(x, 1) for x in g], out_shape=shape, transform=tr, fill=0, dtype=np.uint8)
            if val == 1:
                out[(m == 1) & (out != 255)] = 1
            else:
                out[m == 1] = 255
    return out


def build_model(pretrained: bool = True):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, classes=1)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise(img_u8: np.ndarray) -> np.ndarray:
    return ((img_u8.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def predict(model, img: np.ndarray, device, win: int = 512, overlap: int = 64) -> np.ndarray:
    import torch
    H, W, _ = img.shape
    out = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    # windows overlap, and overlaps get averaged
    step = win - overlap
    ys = list(range(0, max(H - win, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - win, 0) + 1, step)) or [0]
    if ys[-1] + win < H:
        ys.append(H - win)
    if xs[-1] + win < W:
        xs.append(W - win)
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img[y:y + win, x:x + win]
                h, w = tile.shape[:2]
                pad = np.zeros((win, win, 3), dtype=np.uint8)
                pad[:h, :w] = tile
                t = torch.from_numpy(normalise(pad)[None]).to(device)
                p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
                out[y:y + h, x:x + w] += p[:h, :w]
                cnt[y:y + h, x:x + w] += 1
    return out / np.maximum(cnt, 1)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))
