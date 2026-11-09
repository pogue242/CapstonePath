# label candidates against the real gaps
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from missingpaths.geom import aligned_coverage, explode_lines

CAND_STEP_M = 2.0


def _hits_by_owner(cand_geoms: np.ndarray, ref_geoms: np.ndarray, tol, angle, step,
                   chunk: int = 50_000) -> tuple[np.ndarray, pd.DataFrame]:
    parts, owner = explode_lines(ref_geoms)
    tree = shapely.STRtree(parts)
    cov_any = np.zeros(len(cand_geoms))
    frames = []
    for s in range(0, len(cand_geoms), chunk):
        src = cand_geoms[s:s + chunk]
        cov, h = aligned_coverage(src, parts, tol, step, angle, ref_tree=tree, ref_owner=owner)
        cov_any[s:s + len(src)] = cov
        if len(h):
            n_pts = np.maximum(np.ceil(shapely.length(src) / step).astype(int), 2)
            h["cov"] = np.minimum(h["n_points"].to_numpy() / n_pts[h["src"].to_numpy()], 1.0)
            h["src"] += s
            frames.append(h[["src", "ref", "cov"]])
    per = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["src", "ref", "cov"])
    return cov_any, per


def feature_hits(cands: gpd.GeoDataFrame, feats: gpd.GeoDataFrame, tol, angle,
                 step=CAND_STEP_M) -> pd.DataFrame:
    _, per = _hits_by_owner(cands.geometry.to_numpy(), feats.geometry.to_numpy(), tol, angle, step)
    per = per.rename(columns={"src": "cand", "ref": "feat"})
    return per


def label_candidates(cands: gpd.GeoDataFrame, micro_feats: gpd.GeoDataFrame,
                     gip_walk: gpd.GeoDataFrame, osm_original: gpd.GeoDataFrame,
                     tol: float, angle: float, pos_min: float, partial_min: float,
                     step: float = CAND_STEP_M) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits = feature_hits(cands, micro_feats, tol, angle, step)
    tgt = micro_feats["target"].to_numpy()
    h_t = hits[tgt[hits["feat"].to_numpy()]] if len(hits) else hits
    h_s = hits[~tgt[hits["feat"].to_numpy()]] if len(hits) else hits
    n = len(cands)
    best = np.zeros(n)
    best_feat = np.full(n, -1)
    if len(h_t):
        b = h_t.sort_values("cov", ascending=False).drop_duplicates("cand")
        best[b["cand"].to_numpy()] = b["cov"].to_numpy()
        best_feat[b["cand"].to_numpy()] = b["feat"].to_numpy()
    station_cov = np.zeros(n)
    if len(h_s):
        s = h_s.groupby("cand")["cov"].max()
        station_cov[s.index.to_numpy()] = s.to_numpy()
    cg = cands.geometry.to_numpy()
    gip_cov, _ = _hits_by_owner(cg, gip_walk.geometry.to_numpy(), tol, angle, step)
    osm_cov, _ = _hits_by_owner(cg, osm_original.geometry.to_numpy(), tol, angle, step)

    label = np.full(n, "negative", dtype=object)
    reason = np.full(n, "", dtype=object)
    # along a path that already exists: ignored, unless it's a real gap
    ign = [("osm_way_along", osm_cov >= partial_min),
           ("gip_walkable_along", gip_cov >= partial_min),
           ("station_micro", station_cov >= partial_min),
           ("partial_micro", best >= partial_min)]
    for name, m in ign:
        label[m] = "ignore"
        reason[m] = name
    pos = best >= pos_min
    label[pos] = "positive"
    reason[pos] = ""
    fi = np.where(pos, best_feat, -1)
    mf = micro_feats.reset_index(drop=True)
    out = pd.DataFrame({
        "cand_id": cands["cand_id"].to_numpy(),
        "label": label, "ignore_reason": reason,
        "micro_cov": best.round(3), "station_cov": station_cov.round(3),
        "gip_walk_cov": gip_cov.round(3), "osm_orig_cov": osm_cov.round(3),
        "feature_id": np.where(fi >= 0, mf["feature_id"].to_numpy(dtype=object)[np.maximum(fi, 0)], None),
    })
    for col in ("cls", "stratum", "origin"):
        out[f"pos_{col}"] = np.where(fi >= 0, mf[col].to_numpy(dtype=object)[np.maximum(fi, 0)], None)
    return out, hits


def recall_table(feats: pd.DataFrame, hits: pd.DataFrame, pos_min: float, max_euclid: float,
                 group_cols=("cls",)) -> pd.DataFrame:
    f = feats.reset_index(drop=True).copy()
    rec = np.zeros(len(f), dtype=bool)
    if len(hits):
        ok = hits[hits["cov"] >= pos_min]["feat"].unique()
        rec[ok[ok < len(f)]] = True
    f["recalled"] = rec
    f["le_max"] = f["len_m"] <= max_euclid
    rows = []
    for key, grp in f.groupby(list(group_cols)):
        key = key if isinstance(key, tuple) else (key,)
        le = grp[grp["le_max"]]
        rows.append({**dict(zip(group_cols, key)),
                     "features": len(grp), "recalled": int(grp["recalled"].sum()),
                     "recall": round(grp["recalled"].mean(), 3) if len(grp) else np.nan,
                     "features_le_max": len(le),
                     "recall_le_max": round(le["recalled"].mean(), 3) if len(le) else np.nan,
                     "share_longer_than_max": round(1 - grp["le_max"].mean(), 3)})
    return pd.DataFrame(rows)
