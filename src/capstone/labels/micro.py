from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

CLASS_FOW = {
    "passage": [16],
    "stairs": [18, 21],
    "indoor_vertical": [22, 23, 24, 25, 26],
}
FOW_NMT_RAMP = 701
INDOOR_VERTICAL_TOUCH_FOW = {23, 25, 26}


class UnionFind:

    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        p = self.parent
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def component_ids(node_from: np.ndarray, node_to: np.ndarray) -> np.ndarray:
    uf = UnionFind()
    for a, b in zip(node_from.tolist(), node_to.tolist()):
        uf.union(a, b)
    roots = [uf.find(a) for a in node_from.tolist()]
    return pd.factorize(pd.Series(roots))[0]


def micro_links(links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    code2cls = {c: k for k, cs in CLASS_FOW.items() for c in cs}
    fow = pd.to_numeric(links["form_of_way"], errors="coerce")
    m = links[fow.isin(list(code2cls)) & links["walkable"]].copy()
    m["cls"] = pd.to_numeric(m["form_of_way"]).map(code2cls)
    m["feature_id"] = ""
    for cls in CLASS_FOW:
        sub = m[m["cls"] == cls]
        if sub.empty:
            continue
        comp = component_ids(sub["node_from_short_id"].to_numpy(), sub["node_to_short_id"].to_numpy())
        m.loc[sub.index, "feature_id"] = [f"{cls}_{i}" for i in comp]
    return m


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else np.nan


def aggregate_features(ml: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = ml.groupby("feature_id")
    feats = pd.DataFrame({
        "cls": g["cls"].first(),
        "n_links": g.size(),
        "len_m": g["len_m"].sum(),
        "subnet_mode": g["subnet"].agg(_mode),
        "subnets": g["subnet"].agg(lambda s: ",".join(str(int(v)) for v in sorted(set(s.dropna())))),
        "any_level_nonzero": g["level_intermediate"].agg(
            lambda s: bool((pd.to_numeric(s, errors="coerce").fillna(0) != 0).any())),
        "name": g["name_text_hr"].agg(lambda s: _mode(s) if s.notna().any() else ""),
        "regional_code": g["regional_code"].agg(_mode),
    })
    geoms = g.geometry.agg(lambda s: shapely.line_merge(shapely.union_all(s.to_numpy())))
    feats = gpd.GeoDataFrame(feats, geometry=geoms, crs=ml.crs).reset_index()
    return feats


def assign_strata(feats: pd.DataFrame, strata_cfg: dict | None) -> pd.Series:
    out = pd.Series("other", index=feats.index, dtype=object)
    out[feats["cls"] == "stairs"] = "stairs"
    out[feats["cls"] == "indoor_vertical"] = "indoor_vertical"
    passage = feats["cls"] == "passage"
    if strata_cfg:
        sub = pd.to_numeric(feats["subnet_mode"], errors="coerce")
        hist = set(strata_cfg["passage_historic_subnets"])
        est = set(strata_cfg["passage_estate_subnets"])
        out[passage] = "passage_other"
        out[passage & sub.isin(hist)] = "passage_historic"
        out[passage & sub.isin(est)] = "passage_estate"
    else:
        out[passage] = "passage"
    return out
