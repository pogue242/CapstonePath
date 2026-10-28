from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from capstone.labels.micro import INDOOR_VERTICAL_TOUCH_FOW

STATION_AREA_MIN_SHARE = 0.5


def touches_indoor_vertical(micro: gpd.GeoDataFrame, links: gpd.GeoDataFrame) -> pd.Series:
    fow = pd.to_numeric(links["form_of_way"], errors="coerce")
    iv = links[fow.isin(INDOOR_VERTICAL_TOUCH_FOW) & links["walkable"]]
    iv_nodes = set(iv["node_from_short_id"]).union(iv["node_to_short_id"])
    hit = micro["node_from_short_id"].isin(iv_nodes) | micro["node_to_short_id"].isin(iv_nodes)
    return hit.groupby(micro["feature_id"]).any()


def share_inside(feats: gpd.GeoDataFrame, areas: gpd.GeoDataFrame) -> np.ndarray:
    if areas is None or areas.empty:
        return np.zeros(len(feats))
    area = shapely.union_all(areas.geometry.make_valid().to_numpy())
    inside = shapely.length(shapely.intersection(feats.geometry.to_numpy(), area))
    lens = shapely.length(feats.geometry.to_numpy())
    return np.where(lens > 0, inside / np.maximum(lens, 1e-9), 0.0)


def apply_station_filter(feats: gpd.GeoDataFrame, micro: gpd.GeoDataFrame, links: gpd.GeoDataFrame,
                         station_areas: gpd.GeoDataFrame | None, station_subnets: list[int]) -> gpd.GeoDataFrame:
    f = feats.copy()
    f["rule_a_level"] = f["any_level_nonzero"].astype(bool)
    touch = touches_indoor_vertical(micro, links)
    f["rule_b_indoor_vertical"] = f["feature_id"].map(touch).fillna(False).astype(bool)
    f["station_area_share"] = share_inside(f, station_areas)
    f["rule_c_station_area"] = f["station_area_share"] >= STATION_AREA_MIN_SHARE
    f["station_general"] = f["rule_a_level"] | f["rule_b_indoor_vertical"] | f["rule_c_station_area"]
    subs = set(int(s) for s in station_subnets)
    f["station_subnet"] = f["subnets"].map(
        lambda s: bool(subs & {int(x) for x in str(s).split(",") if x.strip().lstrip("-").isdigit()}))
    f["is_station"] = f["station_general"] | f["station_subnet"]
    return f


def agreement_report(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in ["passage", "stairs", "all"]:
        s = f if cls == "all" else f[f["cls"] == cls]
        both = int((s.station_general & s.station_subnet).sum())
        n_sub = int(s.station_subnet.sum())
        n_gen = int(s.station_general.sum())
        rows.append({
            "class": cls, "features": len(s), "subnet_flagged": n_sub, "general_flagged": n_gen,
            "both": both,
            "general_catches_share_of_subnet": round(both / n_sub, 3) if n_sub else np.nan,
            "subnet_confirms_share_of_general": round(both / n_gen, 3) if n_gen else np.nan,
            "rule_a": int(s.rule_a_level.sum()), "rule_b": int(s.rule_b_indoor_vertical.sum()),
            "rule_c": int(s.rule_c_station_area.sum()),
        })
    return pd.DataFrame(rows)
