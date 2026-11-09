# run the trained model on squares and score the results
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import torch

import photo_common as PC


def main(city="vienna", roles="eval,test", plan_file=None, squares_layer="squares.gpkg"):
    plan_file = plan_file or PC.SQUARES_PLAN
    roles = roles.split(",")
    pdir = PC.photo_dir(city)
    odir = PC.run_dir(city) / "prob"
    odir.mkdir(exist_ok=True)
    plan = PC.load_json(pdir / plan_file)
    sq = gpd.read_file(pdir / squares_layer)
    mask = gpd.read_file(pdir / "mask_vectors.gpkg", layer="mask") if squares_layer == "squares.gpkg" else None
    dev = torch.device("cuda")
    model = PC.build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(PC.model_path(city), map_location=dev))
    rows = []
    sq_m = sq.geometry.to_crs(3857)
    for r in sq.itertuples():
        if r.role not in roles:
            continue
        out = odir / f"{r.sq_id}.tif"
        img, bounds = PC.mosaic(city, plan["zoom"], plan["tiles"][r.sq_id])
        if img is None:
            rows.append({"sq_id": r.sq_id, "role": r.role, "status": "missing_tiles"})
            continue
        if out.exists():
            with rasterio.open(out) as src:
                prob = src.read(1).astype(np.float32) / 255
        else:
            prob = PC.predict(model, img, dev)
            tr = PC.transform_for(bounds, prob.shape)
            with rasterio.open(out, "w", driver="GTiff", height=prob.shape[0], width=prob.shape[1], count=1,
                               dtype="uint8", crs="EPSG:3857", transform=tr, compress="deflate") as dst:
                dst.write(np.round(prob * 255).astype(np.uint8), 1)
        row = {"sq_id": r.sq_id, "role": r.role, "status": "ok"}
        if mask is not None:
            m = PC.rasterize_mask(mask, bounds, prob.shape)
            inside = rasterize([(sq_m[r.Index], 1)], out_shape=prob.shape, transform=PC.transform_for(bounds, prob.shape),
                               fill=0, dtype=np.uint8) == 1
            v = (m != 255) & inside
            p, t = prob >= 0.5, m == 1
            row.update(inter=int((p & t & v).sum()), union=int(((p | t) & v).sum()),
                       tp=int((p & t & v).sum()), fp=int((p & ~t & v).sum()), fn=int((~p & t & v).sum()))
        rows.append(row)
        print(r.sq_id, r.role, row.get("inter", ""), row.get("union", ""), flush=True)
    df = pd.DataFrame(rows)
    tag = plan_file.replace(".json", "")
    df.to_csv(PC.ROOT / "reports" / "rebuild" / "tables" / f"{city}_photo_infer_{tag}{('_' + PC.RUN) if PC.RUN else ''}.csv", index=False)
    if "union" in df:
        ok = df[df["status"] == "ok"]
        for role, g in ok.groupby("role"):
            iou = g["inter"].sum() / max(g["union"].sum(), 1)
            prec = g["tp"].sum() / max(g["tp"].sum() + g["fp"].sum(), 1)
            rec = g["tp"].sum() / max(g["tp"].sum() + g["fn"].sum(), 1)
            print(f"ROLE {role}: IoU {iou:.3f} precision {prec:.3f} recall {rec:.3f} squares {len(g)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a)
