# train the U-Net
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F

import photo_common as PC

CROP = 256
BATCH = 8


def prepare(city: str, roles=("seg_train",)):
    pdir = PC.photo_dir(city)
    cache = PC.run_dir(city) / "chips"
    cache.mkdir(exist_ok=True)
    sq = gpd.read_file(pdir / "squares.gpkg", layer="squares")
    plan = PC.load_json(pdir / PC.SQUARES_PLAN)
    mask = gpd.read_file(pdir / "mask_vectors.gpkg", layer="mask")
    out = []
    import os
    layers = [x for x in os.environ.get("PHOTO_TRAIN_LAYERS", PC.LAYER).split(",") if x]
    for layer in layers:
        PC.LAYER = layer
        for r in sq.itertuples():
            if r.role not in roles:
                continue
            f = cache / (f"{r.sq_id}.npz" if len(layers) == 1 else f"{r.sq_id}__{layer}.npz")
            if not f.exists():
                img, bounds = PC.mosaic(city, plan["zoom"], plan["tiles"][r.sq_id])
                if img is None:
                    continue
                m = PC.rasterize_mask(mask, bounds, img.shape[:2])
                np.savez_compressed(f, img=img, mask=m, bounds=np.array(bounds))
            out.append(f)
    return out


def masked_loss(logits, target):
    # 255 means ignore
    valid = (target != 255).float()
    y = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=valid, reduction="sum") / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (y * valid).sum() + 1)
    return bce + dice


def sample_batch(chips, rng):
    xs, ys = [], []
    while len(xs) < BATCH:
        d = chips[rng.integers(len(chips))]
        img, m = d["img"], d["mask"]
        H, W = m.shape
        if H < CROP or W < CROP:
            continue
        y0, x0 = rng.integers(0, H - CROP + 1), rng.integers(0, W - CROP + 1)
        a, b = img[y0:y0 + CROP, x0:x0 + CROP], m[y0:y0 + CROP, x0:x0 + CROP]
        if (b == 1).mean() < 0.002 and rng.random() < 0.7:
            continue
        k = rng.integers(4)
        a, b = np.rot90(a, k), np.rot90(b, k)
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        a = np.clip(a.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        xs.append(PC.normalise(np.ascontiguousarray(a)))
        ys.append(np.ascontiguousarray(b))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)[:, None])


def iou(model, chips, device):
    inter = union = 0
    for d in chips:
        p = PC.predict(model, d["img"], device) >= 0.5
        m = d["mask"]
        v = m != 255
        t = m == 1
        inter += int((p & t & v).sum())
        union += int(((p | t) & v).sum())
    return inter / max(union, 1)


def main(city="vienna", iters=4000):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    files = prepare(city)
    sq_of = [f.stem.split("__")[0] for f in files]
    squares = sorted(set(sq_of))
    rs = np.random.default_rng(42).permutation(len(squares))
    val_sq = {squares[i] for i in rs[:max(3, len(squares) // 10)]}
    val_f = [f for f, s in zip(files, sq_of) if s in val_sq]
    tr_f = [f for f, s in zip(files, sq_of) if s not in val_sq]
    load = lambda fs: [dict(np.load(f)) for f in fs]
    tr, va = load(tr_f), load(val_f)
    print(f"train squares {len(tr)}, val squares {len(va)}", flush=True)
    dev = torch.device("cuda")
    model = PC.build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=iters, pct_start=0.05)
    torch.backends.cudnn.benchmark = True
    ck = PC.run_dir(city) / "seg_ckpt.pt"
    start = 0
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start = s["it"]
        rng = np.random.default_rng(42 + start)
        print(f"resumed at {start}", flush=True)
    logf = PC.ROOT / "reports" / "rebuild" / "tables" / f"{city}_photo_train_log{('_' + PC.RUN) if PC.RUN else ''}.csv"
    new = not logf.exists()
    with open(logf, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["iter", "loss", "val_iou", "seconds"])
        t0, run = time.time(), 0.0
        for it in range(start, iters):
            model.train()
            x, y = sample_batch(tr, rng)
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = masked_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run = 0.98 * run + 0.02 * float(loss) if it > start else float(loss)
            if (it + 1) % 250 == 0 or it + 1 == iters:
                vi = iou(model, va, dev) if (it + 1) % 1000 == 0 or it + 1 == iters else ""
                w.writerow([it + 1, round(run, 4), vi, round(time.time() - t0)])
                fh.flush()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "it": it + 1}, ck)
                print(f"it {it + 1} loss {run:.4f} val_iou {vi} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), PC.run_dir(city) / "seg_final.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vienna", int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
