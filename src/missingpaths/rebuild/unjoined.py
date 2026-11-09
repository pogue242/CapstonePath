# find joining errors and what fixing them saves
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import connected_components, dijkstra

from missingpaths.graph.candidates import EdgeIndex, filter_visible
from missingpaths.graph.distance import NetworkDistance
from missingpaths.graph.walknet import WalkGraph

MAX_GAP_M = 20.0
MAX_ANGLE_DEG = 45.0
MIN_SAVING_M = 25.0
DETOUR_MIN = 2.5
SEARCH_CUTOFF_M = 375.0
BARRIERS = {"fence", "wall", "retaining_wall", "hedge", "city_wall", "guard_rail", "handrail",
                 "jersey_barrier"}


def end_vectors(g: WalkGraph) -> pd.DataFrame:
    geom = g.edges.geometry.to_numpy()
    coords, idx = shapely.get_coordinates(geom, return_index=True)
    first = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    last = np.r_[first[1:] - 1, len(idx) - 1]
    ou = coords[first] - coords[first + 1]
    ov = coords[last] - coords[last - 1]
    e = g.edges
    df = pd.concat([
        pd.DataFrame({"node": e["u"].to_numpy(), "edge": e["edge_id"].to_numpy(), "t": 0.0,
                      "ox": ou[:, 0], "oy": ou[:, 1]}),
        pd.DataFrame({"node": e["v"].to_numpy(), "edge": e["edge_id"].to_numpy(), "t": e["length"].to_numpy(),
                      "ox": ov[:, 0], "oy": ov[:, 1]}),
    ], ignore_index=True)
    n = np.hypot(df["ox"], df["oy"]).replace(0, np.nan)
    df["ox"], df["oy"] = df["ox"] / n, df["oy"] / n
    return df


def find_unjoined(g: WalkGraph, barrier_geoms: np.ndarray | None = None, max_gap: float = MAX_GAP_M,
                  max_angle: float = MAX_ANGLE_DEG, min_saving: float = MIN_SAVING_M,
                  detour_min: float = DETOUR_MIN, cutoff: float = SEARCH_CUTOFF_M):
    eidx = EdgeIndex(g)
    deg = g.nodes["degree"].to_numpy()
    ends = end_vectors(g)
    ends = ends[deg[ends["node"].to_numpy()] == 1].drop_duplicates("node").reset_index(drop=True)
    counts = {"dead_ends": int(len(ends))}
    pts = g.nodes.geometry.to_numpy()[ends["node"].to_numpy()]
    di, ej = eidx.tree.query(pts, predicate="dwithin", distance=max_gap)
    keep = ej != ends["edge"].to_numpy()[di]
    di, ej = di[keep], ej[keep]
    tj = shapely.line_locate_point(eidx.geom[ej], pts[di])
    q = shapely.line_interpolate_point(eidx.geom[ej], tj)
    pxy, qxy = shapely.get_coordinates(pts[di]), shapely.get_coordinates(q)
    d = np.hypot(*(qxy - pxy).T)
    w = (qxy - pxy) / np.maximum(d, 1e-9)[:, None]
    cosang = w[:, 0] * ends["ox"].to_numpy()[di] + w[:, 1] * ends["oy"].to_numpy()[di]
    # the dead end has to point at the other path
    aligned = (d < 1.0) | (cosang >= np.cos(np.radians(max_angle)))
    df = pd.DataFrame({"end": di, "node": ends["node"].to_numpy()[di], "e1": ends["edge"].to_numpy()[di],
                       "t1": ends["t"].to_numpy()[di], "e2": ej, "t2": tj,
                       "x1": pxy[:, 0], "y1": pxy[:, 1], "x2": qxy[:, 0], "y2": qxy[:, 1], "gap_m": d})
    df = df[aligned]
    counts["dead_ends_with_aligned_way_within_gap"] = int(df["end"].nunique())
    df = df.sort_values("gap_m").drop_duplicates("end").reset_index(drop=True)

    lvl = eidx.off_level[df["e1"].to_numpy()] | eidx.off_level[df["e2"].to_numpy()]
    counts["excluded_level_difference"] = int(lvl.sum())
    df = df[~lvl].reset_index(drop=True)

    if barrier_geoms is not None and len(barrier_geoms) and len(df):
        segs = shapely.linestrings(df[["x1", "y1", "x2", "y2"]].to_numpy().reshape(-1, 2, 2))
        btree = shapely.STRtree(np.asarray(barrier_geoms, dtype=object))
        si, _ = btree.query(segs, predicate="intersects")
        bar = np.zeros(len(df), dtype=bool)
        bar[np.unique(si)] = True
    else:
        bar = np.zeros(len(df), dtype=bool)
    counts["excluded_barrier"] = int(bar.sum())
    df = df[~bar].reset_index(drop=True)

    tiny = df["gap_m"] < 0.05
    vis = filter_visible(df[~tiny], eidx)
    df = pd.concat([df[tiny], vis]).sort_index()
    counts["excluded_crosses_other_way"] = int(counts["dead_ends_with_aligned_way_within_gap"]
                                               - counts["excluded_level_difference"]
                                               - counts["excluded_barrier"] - len(df))

    net_dist = NetworkDistance(g, cutoff=cutoff)
    nd = net_dist.point_dist(df["e1"].to_numpy(), df["t1"].to_numpy(), df["e2"].to_numpy(), df["t2"].to_numpy())
    df["netdist"] = nd
    df["saving_m"] = nd - df["gap_m"].to_numpy()
    det = nd / np.maximum(df["gap_m"].to_numpy(), 0.05)
    ok = ~np.isfinite(nd) | ((det >= detour_min) & (df["saving_m"] >= min_saving))
    counts["excluded_already_connected_nearby"] = int((~ok).sum())
    df = df[ok].reset_index(drop=True)

    e = g.edges
    L = e["length"].to_numpy()[df["e2"].to_numpy()]
    at_u = df["t2"].to_numpy() <= 1.0
    at_v = (L - df["t2"].to_numpy()) <= 1.0
    tnode = np.where(at_u, e["u"].to_numpy()[df["e2"].to_numpy()],
                     np.where(at_v, e["v"].to_numpy()[df["e2"].to_numpy()], -1))
    df["target_node"] = tnode
    df["kind"] = np.where((tnode >= 0) & (deg[np.maximum(tnode, 0)] == 1), "end_to_end", "end_to_way")
    a = np.minimum(df["node"], np.where(tnode >= 0, tnode, df["node"]))
    b = np.maximum(df["node"], np.where(tnode >= 0, tnode, df["node"]))
    pair = pd.Series(list(zip(a, b)))
    dup = (df["kind"] == "end_to_end").to_numpy() & pair.duplicated().to_numpy()
    df = df[~dup].reset_index(drop=True)
    counts["unjoined_gaps"] = int(len(df))
    _add_tags(df, g)
    return df, counts


def _way_kind(hw: np.ndarray, fw: np.ndarray, tun: np.ndarray) -> np.ndarray:
    k = np.where(np.isin(hw, ["footway"]), np.where(np.isin(fw, ["sidewalk", "crossing"]), "footway:" + fw, "footway"), hw)
    k = np.where(tun == "building_passage", "building_passage", k)
    road = np.isin(hw, ["residential", "service", "living_street", "unclassified", "tertiary", "secondary",
                        "primary", "tertiary_link", "secondary_link", "primary_link", "road"])
    return np.where(road & (tun != "building_passage"), "road:" + hw, k)


def _add_tags(df: pd.DataFrame, g: WalkGraph) -> None:
    e = g.edges

    def col(c):
        return e[c].fillna("").to_numpy(dtype=object) if c in e else np.full(len(e), "", dtype=object)
    hw, fw, tun = col("highway"), col("footway"), col("tunnel")
    kind = _way_kind(hw, fw, tun)
    df["way_kind"] = kind[df["e1"].to_numpy()]
    df["target_kind"] = kind[df["e2"].to_numpy()]
    df["osm_id_end"] = col("osm_id")[df["e1"].to_numpy()]
    df["osm_id_target"] = col("osm_id")[df["e2"].to_numpy()]


def gaps_gdf(df: pd.DataFrame, crs) -> gpd.GeoDataFrame:
    geom = shapely.linestrings(df[["x1", "y1", "x2", "y2"]].to_numpy().reshape(-1, 2, 2)) if len(df) else []
    return gpd.GeoDataFrame(df, geometry=geom, crs=crs)


def csr_with_links(g: WalkGraph, links: pd.DataFrame) -> sp.csr_matrix:
    base = g.csr()
    if len(links) == 0:
        return base
    e = g.edges
    u, v, L = e["u"].to_numpy(), e["v"].to_numpy(), e["length"].to_numpy()
    e1, t1 = links["e1"].to_numpy(), links["t1"].to_numpy()
    e2, t2 = links["e2"].to_numpy(), links["t2"].to_numpy()
    ln = np.hypot(links["x2"].to_numpy() - links["x1"].to_numpy(), links["y2"].to_numpy() - links["y1"].to_numpy())
    rows, cols, w = [], [], []
    for a, oa in ((u[e1], t1), (v[e1], L[e1] - t1)):
        for b, ob in ((u[e2], t2), (v[e2], L[e2] - t2)):
            rows.append(a)
            cols.append(b)
            w.append(oa + ln + ob)
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.maximum(np.concatenate(w), 1e-3)
    keep = r != c
    r, c, w = r[keep], c[keep], w[keep]
    bc = base.tocoo()
    df = pd.DataFrame({"r": np.r_[bc.row, r, c], "c": np.r_[bc.col, c, r], "w": np.r_[bc.data, w, w]})
    df = df.groupby(["r", "c"], as_index=False)["w"].min()
    return sp.csr_matrix((df["w"].to_numpy(), (df["r"].to_numpy(), df["c"].to_numpy())), shape=base.shape)


def n_components(csr: sp.csr_matrix, node_mask: np.ndarray | None = None) -> int:
    _, lab = connected_components(csr, directed=False)
    has = np.diff(csr.indptr) > 0
    if node_mask is not None:
        has &= node_mask
    return int(len(np.unique(lab[has])))


def trip_impact(g: WalkGraph, csr_before: sp.csr_matrix, csr_after: sp.csr_matrix, origins: np.ndarray,
                seed: int, dmin: float = 200.0, dmax: float = 2000.0, limit: float = 6000.0,
                chunk: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xy = shapely.get_coordinates(g.nodes.geometry.to_numpy())
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    rows = []
    origins = np.asarray(origins)
    dests = np.full(len(origins), -1)
    for i, o in enumerate(origins):
        cand = tree.query_ball_point(xy[o], dmax)
        cand = [c for c in cand if np.hypot(*(xy[c] - xy[o])) >= dmin]
        if cand:
            dests[i] = rng.choice(cand)
    ok = dests >= 0
    origins, dests = origins[ok], dests[ok]
    for s in range(0, len(origins), chunk):
        o, d = origins[s:s + chunk], dests[s:s + chunk]
        db = dijkstra(csr_before, directed=False, indices=o, limit=limit)[np.arange(len(o)), d]
        da = dijkstra(csr_after, directed=False, indices=o, limit=limit)[np.arange(len(o)), d]
        rows.append(pd.DataFrame({"origin": o, "dest": d, "euclid": np.hypot(*(xy[d] - xy[o]).T),
                                  "before_m": db, "after_m": da}))
    out = pd.concat(rows, ignore_index=True)
    out["changed"] = out["after_m"] < out["before_m"] - 1.0
    out["newly_reachable"] = ~np.isfinite(out["before_m"]) & np.isfinite(out["after_m"])
    out["saved_m"] = np.where(np.isfinite(out["before_m"]), out["before_m"] - out["after_m"], np.nan)
    return out
