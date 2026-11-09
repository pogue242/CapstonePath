# turn the heat map into candidate lines
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.features import rasterize
from skimage.measure import label as cc_label
from skimage.morphology import remove_small_objects, skeletonize

import photo_common as PC
from missingpaths.labels.candidates_label import feature_hits

CRS = "EPSG:31256"
SCALE = 1.0 / np.cos(np.radians(48.2))


def trace_square(path, edges_merc, to_metric):
    with rasterio.open(path) as src:
        p = src.read(1).astype(np.float32) / 255
        tr = src.transform
    if tr.a < 0.2:
        H, W = (p.shape[0] // 2) * 2, (p.shape[1] // 2) * 2
        p = p[:H, :W].reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
        tr = tr * tr.scale(2, 2)
    px = tr.a
    # keep pixels at 0.5 or more, then thin to lines
    walk = remove_small_objects(p >= 0.5, min_size=int(8 * SCALE ** 2 / px ** 2 * 10))
    sk = skeletonize(walk)
    # remove anything within 3 m of OSM
    near = rasterize([(g, 1) for g in shapely.buffer(edges_merc, 3.0 * SCALE)], out_shape=sk.shape,
                     transform=tr, fill=0, dtype=np.uint8) if len(edges_merc) else np.zeros(sk.shape, np.uint8)
    sk &= near == 0
    lab = cc_label(sk, connectivity=2)
    segs = []
    for k in range(1, lab.max() + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) * px / SCALE < 8.0:
            continue
        x, y = rasterio.transform.xy(tr, rr, cc)
        mx, my = to_metric.transform(np.asarray(x), np.asarray(y))
        xy = np.c_[mx, my]
        c = xy - xy.mean(axis=0)
        ax = np.linalg.svd(c, full_matrices=False)[2][0]
        tproj = c @ ax
        lo, hi = tproj.min(), tproj.max()
        nb = max(1, int(np.ceil((hi - lo) / 30.0)))
        edges = np.linspace(lo, hi, nb + 1)
        for b in range(nb):
            m = (tproj >= edges[b]) & (tproj <= edges[b + 1])
            if m.sum() < 2:
                continue
            a, z = xy[m][np.argmin(tproj[m])], xy[m][np.argmax(tproj[m])]
            if np.hypot(*(z - a)) >= 6.0:
                segs.append(("P_trace", a, z))
        segs.append(("P_end", xy[np.argmin(tproj)], None))
        segs.append(("P_end", xy[np.argmax(tproj)], None))
    return segs


def main(city="vienna"):
    pdir = PC.photo_dir(city)
    root = PC.ROOT / "data" / "interim" / city
    sq = gpd.read_file(pdir / "squares.gpkg")
    ev = sq[sq["role"] == "eval"]
    edges = gpd.read_file(root / "walk_graph_original.gpkg", layer="edges")
    etree = shapely.STRtree(edges.geometry.to_numpy())
    to_m = Transformer.from_crs("EPSG:3857", CRS, always_xy=True)
    edges_merc = edges.to_crs(3857)
    rows = []
    for r in ev.itertuples():
        f = PC.run_dir(city) / "prob" / f"{r.sq_id}.tif"
        if not f.exists():
            continue
        box_merc = gpd.GeoSeries([r.geometry], crs=CRS).to_crs(3857).iloc[0]
        loc = edges_merc.geometry.to_numpy()[shapely.STRtree(edges_merc.geometry.to_numpy()).query(box_merc.buffer(20))]
        for kind, a, z in trace_square(f, loc, to_m):
            if kind == "P_trace":
                rows.append({"sq_id": r.sq_id, "gen": "P_trace", "geometry": shapely.LineString([a, z])})
            else:
                pt = shapely.Point(a)
                j = etree.query_nearest(pt, max_distance=30.0, all_matches=False)
                if len(j):
                    q = shapely.ops.nearest_points(pt, edges.geometry.iat[int(j[0])])[1]
                    if 2.0 <= pt.distance(q) <= 30.0:
                        rows.append({"sq_id": r.sq_id, "gen": "P_conn", "geometry": shapely.LineString([pt, q])})
        print(r.sq_id, len(rows), flush=True)
    pc = gpd.GeoDataFrame(rows, crs=CRS)
    pc.to_file(PC.run_dir(city) / "photo_candidates.gpkg", layer="photo_candidates", driver="GPKG")
    gaps = gpd.read_file(root / "rebuild" / "real_gaps.gpkg", layer="real_gaps")
    centre = gaps[gaps["gap_id"].isin(ev["centre_gap"].dropna())].reset_index(drop=True)
    inside = gaps[gaps.within(shapely.union_all(ev.geometry.to_numpy()))].reset_index(drop=True)
    cands = gpd.read_parquet(root / "rebuild" / "candidates.parquet")
    out = []
    for setname, gs in (("centre_sample", centre), ("all_inside_eval_squares", inside)):
        near = np.unique(cands.sindex.query(gs.geometry.buffer(10), predicate="intersects")[1])
        sub = cands.iloc[near].reset_index(drop=True)
        hm = feature_hits(sub, gs, 10, 35)
        hm = hm[hm["cov"] >= 0.8]
        reached_map = set(hm["feat"])
        reached_abc = set(hm[hm["cand"].isin(np.flatnonzero(sub[["gen_A", "gen_B", "gen_C"]].any(axis=1)))]["feat"])
        hp = feature_hits(pc, gs, 10, 35) if len(pc) else pd.DataFrame(columns=["cand", "feat", "cov"])
        reached_p = set(hp[hp["cov"] >= 0.8]["feat"])
        for grp in ["all"] + sorted(gs["group"].unique()):
            idx = set(range(len(gs))) if grp == "all" else set(np.flatnonzero(gs["group"].to_numpy() == grp))
            n = len(idx)
            out.append({"gap_set": setname, "group": grp, "gaps": n,
                        "reach_A_B_C": round(len(idx & reached_abc) / max(n, 1), 3),
                        "reach_map_A_B_C_E": round(len(idx & reached_map) / max(n, 1), 3),
                        "reach_photo_P": round(len(idx & reached_p) / max(n, 1), 3),
                        "reach_map_plus_photo": round(len(idx & (reached_map | reached_p)) / max(n, 1), 3)})
    df = pd.DataFrame(out)
    df.to_csv(PC.ROOT / "reports" / "rebuild" / "tables" / f"{city}_reach_photo_eval{('_' + PC.RUN) if PC.RUN else ''}.csv", index=False)
    print(df.to_string(), flush=True)
    print(f"photo candidates: {pc['gen'].value_counts().to_dict() if len(pc) else 0}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    import shapely.ops
    main(sys.argv[1] if len(sys.argv) > 1 else "vienna")
