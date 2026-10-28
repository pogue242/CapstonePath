from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from capstone.geom import aligned_coverage, explode_lines, match_status

REVIEW_STRATA = {"passage_historic": 30, "passage_estate": 30, "stairs": 30}
MICRO_TAG_HIGHWAYS = {"steps"}
KEEP_ROAD_HIGHWAYS = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
                      "unclassified", "residential", "living_street", "service", "trunk", "trunk_link",
                      "motorway", "motorway_link", "road", "busway"}
DROP_WHOLE_WAY_SHARE = 0.5


def match_features(feats: gpd.GeoDataFrame, osm_ways: gpd.GeoDataFrame, tol: float, angle: float,
                   step: float, matched_min: float, partial_min: float, tree=None):
    parts, owner = explode_lines(feats.geometry.to_numpy())
    ref = osm_ways.geometry.to_numpy()
    cov_p, hits = aligned_coverage(parts, ref, tol, step, angle, ref_tree=tree)
    lens = shapely.length(parts)
    wc = np.bincount(owner, weights=cov_p * lens, minlength=len(feats))
    tl = np.bincount(owner, weights=lens, minlength=len(feats))
    cov = np.where(tl > 0, wc / np.maximum(tl, 1e-9), 0.0)
    out = feats[["feature_id"]].copy()
    out["coverage"] = cov.round(3)
    out["status"] = match_status(cov, matched_min, partial_min)
    hits = hits.assign(feature=owner[hits["src"].to_numpy()]) if len(hits) else hits.assign(feature=[])
    hits = hits.groupby(["feature", "ref"], as_index=False)["n_points"].sum()
    return out, hits


def dominant_way(hits: pd.DataFrame, n: int) -> np.ndarray:
    dom = np.full(n, -1)
    if len(hits):
        h = hits.sort_values("n_points", ascending=False).drop_duplicates("feature")
        dom[h["feature"].to_numpy()] = h["ref"].to_numpy()
    return dom


def reverse_match(osm_micro: gpd.GeoDataFrame, gip_walk: gpd.GeoDataFrame, tol, angle, step,
                  matched_min, partial_min) -> pd.DataFrame:
    parts, owner = explode_lines(osm_micro.geometry.to_numpy())
    cov_p, _ = aligned_coverage(parts, gip_walk.geometry.to_numpy(), tol, step, angle)
    lens = shapely.length(parts)
    wc = np.bincount(owner, weights=cov_p * lens, minlength=len(osm_micro))
    tl = np.bincount(owner, weights=lens, minlength=len(osm_micro))
    cov = np.where(tl > 0, wc / np.maximum(tl, 1e-9), 0.0)
    out = osm_micro[["osm_id", "kind"]].copy()
    out["coverage"] = cov.round(3)
    out["status"] = match_status(cov, matched_min, partial_min, ("in_gip", "partial", "osm_only"))
    return out


def deletion_plan(feats: gpd.GeoDataFrame, status: pd.DataFrame, hits: pd.DataFrame,
                  osm_ways: gpd.GeoDataFrame, tol: float) -> pd.DataFrame:
    sel = status["status"].isin(["matched", "partial"]).to_numpy() & feats["target"].to_numpy()
    feat_idx = np.flatnonzero(sel)
    h = hits[hits["feature"].isin(feat_idx)]
    rows = []
    geoms = osm_ways.geometry.to_numpy()
    fgeo = feats.geometry.to_numpy()
    hw = osm_ways["highway"].to_numpy()
    tun = osm_ways["tunnel"].to_numpy() if "tunnel" in osm_ways else np.full(len(osm_ways), None)
    cov = osm_ways["covered"].to_numpy() if "covered" in osm_ways else np.full(len(osm_ways), None)
    for f, r in zip(h["feature"].to_numpy(), h["ref"].to_numpy()):
        g = geoms[r]
        buf = shapely.buffer(fgeo[f], tol)
        share = shapely.length(shapely.intersection(g, buf)) / max(shapely.length(g), 1e-9)
        micro_tag = (hw[r] in MICRO_TAG_HIGHWAYS or tun[r] == "building_passage"
                     or cov[r] in ("yes", "arcade"))
        if hw[r] in KEEP_ROAD_HIGHWAYS and tun[r] != "building_passage":
            action = "keep_road"
        else:
            action = "drop" if (share >= DROP_WHOLE_WAY_SHARE or micro_tag) else "cut"
        rows.append({"way_index": int(r), "feature_id": feats["feature_id"].iat[f],
                     "action": action, "share_on_feature": round(float(share), 3)})
    return pd.DataFrame(rows, columns=["way_index", "feature_id", "action", "share_on_feature"])


def apply_deletion(osm_ways: gpd.GeoDataFrame, plan: pd.DataFrame, feats: gpd.GeoDataFrame,
                   tol: float) -> gpd.GeoDataFrame:
    ways = osm_ways.copy()
    if plan.empty:
        return ways
    drop = set(plan.loc[plan["action"] == "drop", "way_index"])
    cuts = plan[(plan["action"] == "cut") & ~plan["way_index"].isin(drop)]
    fgeom = feats.set_index("feature_id").geometry
    geoms = ways.geometry.to_numpy().copy()
    for w, grp in cuts.groupby("way_index"):
        buf = shapely.union_all([shapely.buffer(fgeom[f], tol) for f in grp["feature_id"]])
        rest = shapely.difference(geoms[w], buf)
        pieces = [p for p in shapely.get_parts(shapely.line_merge(rest)) if p.length >= 1.0] \
            if not rest.is_empty else []
        geoms[w] = shapely.multilinestrings(pieces) if pieces else None
    ways = ways.set_geometry(gpd.GeoSeries(geoms, index=ways.index, crs=ways.crs))
    keep = np.ones(len(ways), dtype=bool)
    keep[sorted(drop)] = False
    ways = ways[keep]
    return ways[ways.geometry.notna() & ~ways.geometry.is_empty]


def stratified_review_sample(df: pd.DataFrame, seed: int, crs_metric) -> pd.DataFrame:
    pool = df[(df["status"] == "gip_only") & ~df["is_station"]].sort_values("feature_id")
    rows = []
    for stratum, n in REVIEW_STRATA.items():
        s = pool[pool["stratum"] == stratum]
        take = s.sample(n=min(n, len(s)), random_state=seed) if len(s) else s
        rows.append(take)
    sample = pd.concat(rows)
    sample = gpd.GeoDataFrame(sample, geometry="geometry", crs=crs_metric)
    c = sample.geometry.representative_point().to_crs(4326)
    lat, lon = c.y.round(6).astype(str), c.x.round(6).astype(str)
    out = pd.DataFrame({
        "review_id": np.arange(1, len(sample) + 1),
        "feature_id": sample["feature_id"].to_numpy(),
        "stratum": sample["stratum"].to_numpy(),
        "len_m": sample["len_m"].round(1).to_numpy(),
        "coverage": sample["coverage"].to_numpy(),
        "name": sample["name"].fillna("").to_numpy(),
        "lat": lat.to_numpy(), "lon": lon.to_numpy(),
        "osm_link": ("https://www.openstreetmap.org/?mlat=" + lat + "&mlon=" + lon + "#map=19/" + lat + "/" + lon).to_numpy(),
        "map_link": ("https://www.mapillary.com/app/?lat=" + lat + "&lng=" + lon + "&z=18").to_numpy(),
        "reviewer_a": "", "reviewer_b": "", "final": "", "final_reason": "",
    })
    return out


def sensitivity_table(feats: gpd.GeoDataFrame, refs: dict[str, gpd.GeoDataFrame], tols, angles,
                      step, matched_min, partial_min) -> pd.DataFrame:
    rows = []
    for ref_name, ref in refs.items():
        tree = shapely.STRtree(ref.geometry.to_numpy())
        for tol in tols:
            for ang in angles:
                st, _ = match_features(feats, ref, tol, ang, step, matched_min, partial_min, tree=tree)
                t = pd.crosstab(feats["cls"].to_numpy(), st["status"].to_numpy())
                for cls, r in t.iterrows():
                    rows.append({"reference": ref_name, "tol_m": tol, "angle_deg": ang, "class": cls,
                                 **{k: int(r.get(k, 0)) for k in ("matched", "partial", "gip_only")}})
    return pd.DataFrame(rows)
