# sort GIP links into what OSM has and what it is missing
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from missingpaths.geom import aligned_coverage, explode_lines
from missingpaths.labels.micro import UnionFind

STAIRS_FOW = {18, 21}
PASSAGE_FOW = {16}
INDOOR_FOW = {22, 23, 24, 25, 26}
GROUP_OF_TYPE = {"footpath": "footpath", "passage_estate": "estate_passage",
                 "passage_historic": "historic_passage", "passage_other": "other",
                 "stairs": "stairs", "street": "street", "indoor": "indoor"}
GAP_GROUPS = ["footpath", "estate_passage", "historic_passage", "stairs", "other"]
ROAD_HIGHWAYS = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
                 "unclassified", "residential", "living_street", "service", "road", "trunk", "trunk_link"}
SIDEWALK_TOL_M = 15.0
PLAZA_MIN_SHARE = 0.5
STATION_AREA_MIN_SHARE = 0.5
MIN_GAP_LEN_M = 5.0


def link_types(links: pd.DataFrame, strata: dict, station_subnets) -> pd.DataFrame:
    fow = pd.to_numeric(links["form_of_way"], errors="coerce").fillna(-1).astype(int)
    sub = pd.to_numeric(links["subnet"], errors="coerce").fillna(-1).astype(int)
    t = np.where(links["car"].to_numpy(), "street", "footpath").astype(object)
    pas = fow.isin(PASSAGE_FOW).to_numpy()
    t[pas] = "passage_other"
    t[pas & sub.isin(strata["passage_historic_subnets"]).to_numpy()] = "passage_historic"
    t[pas & sub.isin(strata["passage_estate_subnets"]).to_numpy()] = "passage_estate"
    t[fow.isin(STAIRS_FOW).to_numpy()] = "stairs"
    t[fow.isin(INDOOR_FOW).to_numpy()] = "indoor"
    out = pd.DataFrame(index=links.index)
    out["ltype"] = t
    out["group"] = pd.Series(t, index=links.index).map(GROUP_OF_TYPE)
    lvl = pd.to_numeric(links["level_intermediate"], errors="coerce").fillna(0)
    out["st_level"] = (lvl != 0).to_numpy()
    out["st_subnet"] = sub.isin(list(station_subnets)).to_numpy()
    out["st_indoor_fow"] = fow.isin(INDOOR_FOW).to_numpy()
    return out


def share_inside(geoms: np.ndarray, polys: np.ndarray) -> np.ndarray:
    n = len(geoms)
    if len(polys) == 0 or n == 0:
        return np.zeros(n)
    polys = shapely.make_valid(np.asarray(polys, dtype=object))
    tree = shapely.STRtree(polys)
    si, pj = tree.query(geoms, predicate="intersects")
    if len(si) == 0:
        return np.zeros(n)
    ln = shapely.length(shapely.intersection(geoms[si], polys[pj]))
    tot = np.bincount(si, weights=ln, minlength=n)
    return np.minimum(tot / np.maximum(shapely.length(geoms), 1e-9), 1.0)


def coverage(lines: np.ndarray, ref: np.ndarray, tol: float, angle: float, step: float,
             chunk: int = 20_000, tree: shapely.STRtree | None = None) -> np.ndarray:
    parts, owner = explode_lines(np.asarray(lines, dtype=object))
    ref = np.asarray(ref, dtype=object)
    tree = tree or shapely.STRtree(ref)
    cov_p = np.zeros(len(parts))
    for s in range(0, len(parts), chunk):
        c, _ = aligned_coverage(parts[s:s + chunk], ref, tol, step, angle, ref_tree=tree)
        cov_p[s:s + chunk] = c
    lens = shapely.length(parts)
    wc = np.bincount(owner, weights=cov_p * lens, minlength=len(lines))
    tl = np.bincount(owner, weights=lens, minlength=len(lines))
    return np.where(tl > 0, wc / np.maximum(tl, 1e-9), 0.0)


def status_of(cov: np.ndarray, in_min: float = 0.8, partly_min: float = 0.3) -> np.ndarray:
    return np.where(cov >= in_min, "in_osm", np.where(cov >= partly_min, "partly", "missing"))


def classify(df: pd.DataFrame) -> np.ndarray:
    c = np.full(len(df), "", dtype=object)
    st = df["status"].to_numpy()
    c[st == "in_osm"] = "in_osm"
    off = st != "in_osm"
    rules = [
        ("station_or_indoor", df["is_station"].to_numpy()),
        ("sidewalk_drawn_differently", df["sidewalk_like"].to_numpy()),
        ("plaza_drawn_as_area", df["plaza_like"].to_numpy()),
        ("unjoined_in_osm", df["unjoined_like"].to_numpy()),
    ]
    for name in ("cemetery_path", "rural_track"):
        if f"{name}_like" in df:
            rules.append((name, df[f"{name}_like"].to_numpy()))
    for name, m in rules:
        sel = off & (c == "") & m
        c[sel] = name
    rest = off & (c == "")
    c[rest] = np.where(st[rest] == "partly", "partly", "missing")
    return c


CEMETERY = {"landuse": ["cemetery"], "amenity": ["grave_yard"]}
RURAL = {"landuse": ["farmland", "meadow", "vineyard", "orchard", "farmyard", "greenhouse_horticulture", "forest"],
         "natural": ["wood", "scrub", "grassland", "heath"]}
LANDUSE_MIN_SHARE = 0.5


def landuse_polys(areas: pd.DataFrame, spec: dict) -> np.ndarray:
    from missingpaths.features.context import subdivide
    m = pd.Series(False, index=areas.index)
    for col, vals in spec.items():
        if col in areas:
            m |= areas[col].isin(vals)
    return subdivide(areas.loc[m].geometry.to_numpy())


def merge_gap_links(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    feats = []
    for grp, sub in df.groupby("group"):
        uf = UnionFind()
        for a, b in zip(sub["node_from_short_id"].tolist(), sub["node_to_short_id"].tolist()):
            uf.union(a, b)
        roots = [uf.find(a) for a in sub["node_from_short_id"].tolist()]
        comp = pd.factorize(pd.Series(roots, index=sub.index))[0]
        s = sub.assign(_c=comp)
        g = s.groupby("_c")
        geom = g.geometry.agg(lambda x: shapely.line_merge(shapely.union_all(x.to_numpy())))
        f = pd.DataFrame({
            "group": grp,
            "ltype": g["ltype"].agg(lambda x: x.value_counts().index[0]),
            "n_links": g.size(), "len_m": g["len_m"].sum(),
            "cov_mean": g.apply(lambda x: float(np.average(x["cov8"], weights=np.maximum(x["len_m"], 1e-6))),
                                include_groups=False),
            "link_ids": g["object_id"].agg(lambda x: ",".join(map(str, x))) if "object_id" in s else "",
        })
        feats.append(gpd.GeoDataFrame(f, geometry=geom.to_numpy(), crs=df.crs))
    out = pd.concat(feats, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=df.crs)
