# make candidate links from the OSM graph
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from missingpaths.geom import angle_diff, bearing_at, segment_bearing
from missingpaths.graph.distance import NetworkDistance
from missingpaths.graph.walknet import WalkGraph

ROAD_HIGHWAYS = {"primary", "primary_link", "secondary", "secondary_link", "tertiary",
                 "tertiary_link", "unclassified", "residential", "living_street", "service",
                 "trunk", "trunk_link", "motorway", "motorway_link", "road", "busway"}
SIDEWALK_FOOTWAYS = {"sidewalk", "crossing", "traffic_island"}


@dataclass
class GenParams:
    min_euclid: float = 8.0
    max_euclid: float = 150.0
    detour_min: float = 2.5
    step: float = 10.0
    dedupe_m: float = 10.0
    stairs_max_euclid: float = 60.0
    parallel_deg: float = 30.0
    min_face_area: float = 30.0
    cutoff: float | None = None

    @classmethod
    def from_config(cls, p: dict) -> "GenParams":
        c = p["candidates"]
        return cls(min_euclid=c["min_euclid_m"], max_euclid=c["max_euclid_m"],
                   detour_min=c["detour_ratio_min"], step=c["block_perimeter_step_m"],
                   dedupe_m=c["dedupe_endpoint_m"],
                   stairs_max_euclid=c.get("stairs_max_euclid_m", 60.0),
                   parallel_deg=c.get("parallel_angle_deg", 30.0),
                   min_face_area=c.get("min_face_area_m2", 30.0))

    @property
    def search_cutoff(self) -> float:
        return self.cutoff or self.detour_min * self.max_euclid


class EdgeIndex:

    def __init__(self, g: WalkGraph):
        e = g.edges
        self.g = g
        self.geom = e.geometry.to_numpy()
        self.tree = shapely.STRtree(self.geom)
        hw = e["highway"].fillna("").to_numpy(dtype=object)
        fw = (e["footway"] if "footway" in e else pd.Series("", index=e.index)).fillna("").to_numpy(dtype=object)
        self.is_road = np.isin(hw, list(ROAD_HIGHWAYS))
        self.is_sidewalk = np.isin(fw, list(SIDEWALK_FOOTWAYS))
        lvl = np.zeros(len(e), dtype=bool)
        for col in ("bridge", "tunnel"):
            if col in e:
                v = e[col].fillna("no").to_numpy(dtype=object)
                lvl |= ~np.isin(v, ["no", ""])
        if "layer" in e:
            layer = pd.to_numeric(e["layer"], errors="coerce").fillna(0).to_numpy()
            lvl |= layer != 0
        self.off_level = lvl
        self.blockers = np.flatnonzero(~lvl)
        self.block_tree = shapely.STRtree(self.geom[self.blockers])

    def locate(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self.tree.query_nearest(pts, return_distance=False, all_matches=False)
        e = idx[1][np.argsort(idx[0])]
        t = shapely.line_locate_point(self.geom[e], pts)
        return e, t

    def bearing(self, e: np.ndarray, t: np.ndarray) -> np.ndarray:
        return bearing_at(self.geom[e], t, h=2.0)


def _pairs_frame(e1, t1, e2, t2, eidx: EdgeIndex, gen: str) -> pd.DataFrame:
    p1 = shapely.line_interpolate_point(eidx.geom[e1], t1)
    p2 = shapely.line_interpolate_point(eidx.geom[e2], t2)
    x1, y1 = shapely.get_x(p1), shapely.get_y(p1)
    x2, y2 = shapely.get_x(p2), shapely.get_y(p2)
    return pd.DataFrame({"e1": e1, "t1": t1, "e2": e2, "t2": t2, "x1": x1, "y1": y1,
                         "x2": x2, "y2": y2, "euclid": np.hypot(x2 - x1, y2 - y1), "gen": gen})


def filter_length(df: pd.DataFrame, p: GenParams, max_euclid: float | None = None) -> pd.DataFrame:
    hi = p.max_euclid if max_euclid is None else max_euclid
    return df[(df["euclid"] >= p.min_euclid) & (df["euclid"] <= hi) & (df["e1"] != df["e2"])]


def filter_sidewalk_road(df: pd.DataFrame, eidx: EdgeIndex, p: GenParams) -> pd.DataFrame:
    e1, e2 = df["e1"].to_numpy(), df["e2"].to_numpy()
    mixed = (eidx.is_sidewalk[e1] & eidx.is_road[e2]) | (eidx.is_road[e1] & eidx.is_sidewalk[e2])
    if not mixed.any():
        return df
    b1 = eidx.bearing(e1, df["t1"].to_numpy())
    b2 = eidx.bearing(e2, df["t2"].to_numpy())
    parallel = angle_diff(b1, b2) <= p.parallel_deg
    return df[~(mixed & parallel)]


VISIBILITY_TRIM_M = 0.05


def filter_visible(df: pd.DataFrame, eidx: EdgeIndex, chunk: int = 500_000) -> pd.DataFrame:
    if df.empty:
        return df
    keep = np.ones(len(df), dtype=bool)
    xy = df[["x1", "y1", "x2", "y2"]].to_numpy()
    d = xy[:, 2:] - xy[:, :2]
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    f = np.minimum(VISIBILITY_TRIM_M / L, 0.49)
    trimmed = np.hstack([xy[:, :2] + d * f, xy[:, 2:] - d * f])
    for s in range(0, len(df), chunk):
        seg = shapely.linestrings(trimmed[s:s + chunk].reshape(-1, 2, 2))
        qi, _ = eidx.block_tree.query(seg, predicate="intersects")
        keep[s + np.unique(qi)] = False
    return df[keep]


def add_network_distance(df: pd.DataFrame, net_dist: NetworkDistance) -> pd.DataFrame:
    df = df.copy()
    nd = net_dist.point_dist(df["e1"].to_numpy(), df["t1"].to_numpy(),
                           df["e2"].to_numpy(), df["t2"].to_numpy())
    df["netdist"] = nd
    df["detour"] = nd / np.maximum(df["euclid"].to_numpy(), 1e-6)
    return df


def filter_detour(df: pd.DataFrame, p: GenParams) -> pd.DataFrame:
    return df[df["detour"] >= p.detour_min]


def generator_a(eidx: EdgeIndex, p: GenParams) -> pd.DataFrame:
    g = eidx.g
    dead = g.nodes.index[g.nodes["degree"].to_numpy() == 1].to_numpy()
    if len(dead) == 0:
        return _pairs_frame(*(np.zeros(0, int),) * 4, eidx, "A").iloc[:0]
    dpts = g.nodes.geometry.to_numpy()[dead]
    e = g.edges
    own = pd.concat([pd.DataFrame({"node": e["u"], "edge": e["edge_id"], "t": 0.0}),
                     pd.DataFrame({"node": e["v"], "edge": e["edge_id"], "t": e["length"]})])
    own = own.drop_duplicates("node").set_index("node")
    de, dt = own.loc[dead, "edge"].to_numpy(), own.loc[dead, "t"].to_numpy()
    di, ej = eidx.tree.query(dpts, predicate="dwithin", distance=p.max_euclid)
    tj = shapely.line_locate_point(eidx.geom[ej], dpts[di])
    df = _pairs_frame(de[di], dt[di], ej, tj, eidx, "A")
    return df


def block_faces(eidx: EdgeIndex, p: GenParams) -> np.ndarray:
    noded = shapely.union_all(eidx.geom)
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
    return faces[shapely.area(faces) >= p.min_face_area]


def generator_b(eidx: EdgeIndex, p: GenParams, faces: np.ndarray | None = None) -> pd.DataFrame:
    if faces is None:
        faces = block_faces(eidx, p)
    rings = []
    owner = []
    for i, f in enumerate(faces):
        rings.append(shapely.get_exterior_ring(f))
        owner.append(i)
        for k in range(shapely.get_num_interior_rings(f)):
            rings.append(shapely.get_interior_ring(f, k))
            owner.append(i)
    rings = np.asarray(rings, dtype=object)
    owner = np.asarray(owner)
    lens = shapely.length(rings)
    n = np.maximum(np.floor(lens / p.step).astype(int), 1)
    ring_of_pt = np.repeat(np.arange(len(rings)), n)
    frac = np.concatenate([np.arange(k) / k for k in n])
    pts = shapely.line_interpolate_point(rings[ring_of_pt], frac * lens[ring_of_pt])
    face_of_pt = owner[ring_of_pt]
    pe, pt_ = eidx.locate(pts)
    xy = shapely.get_coordinates(pts)
    parts = []
    order = np.argsort(face_of_pt, kind="stable")
    brk = np.flatnonzero(np.r_[True, np.diff(face_of_pt[order]) != 0, True])
    for s, e in zip(brk[:-1], brk[1:]):
        ids = order[s:e]
        if len(ids) < 2:
            continue
        tree = cKDTree(xy[ids])
        pr = tree.query_pairs(p.max_euclid, output_type="ndarray")
        if len(pr) == 0:
            continue
        i, j = ids[pr[:, 0]], ids[pr[:, 1]]
        parts.append((i, j))
    if not parts:
        return _pairs_frame(*(np.zeros(0, int),) * 4, eidx, "B").iloc[:0]
    i = np.concatenate([a for a, _ in parts])
    j = np.concatenate([b for _, b in parts])
    df = _pairs_frame(pe[i], pt_[i], pe[j], pt_[j], eidx, "B")
    return df


def generator_c(eidx: EdgeIndex, p: GenParams) -> pd.DataFrame:
    geom = eidx.geom
    lens = shapely.length(geom)
    n = np.maximum(np.floor(lens / p.step).astype(int), 1)
    e_of = np.repeat(np.arange(len(geom)), n)
    frac = (np.concatenate([np.arange(k) for k in n]) + 0.5) / n[e_of]
    t = frac * lens[e_of]
    pts = shapely.line_interpolate_point(geom[e_of], t)
    b_src = eidx.bearing(e_of, t)
    pi, ej = eidx.tree.query(pts, predicate="dwithin", distance=p.stairs_max_euclid)
    keep = ej != e_of[pi]
    pi, ej = pi[keep], ej[keep]
    tj = shapely.line_locate_point(geom[ej], pts[pi])
    par = angle_diff(b_src[pi], eidx.bearing(ej, tj)) <= p.parallel_deg
    pi, ej, tj = pi[par], ej[par], tj[par]
    df = _pairs_frame(e_of[pi], t[pi], ej, tj, eidx, "C")
    seg_b = segment_bearing(df["x1"], df["y1"], df["x2"], df["y2"])
    df = df[angle_diff(seg_b, b_src[pi]) >= 90 - p.parallel_deg]
    return df


def order_ends(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    swap = (df["x1"] > df["x2"]) | ((df["x1"] == df["x2"]) & (df["y1"] > df["y2"]))
    for a, b in (("x1", "x2"), ("y1", "y2"), ("e1", "e2"), ("t1", "t2")):
        va, vb = df[a].to_numpy().copy(), df[b].to_numpy().copy()
        df.loc[swap, a] = vb[swap.to_numpy()]
        df.loc[swap, b] = va[swap.to_numpy()]
    return df


def endpoint_nms(df: pd.DataFrame, radius: float, priority: np.ndarray) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    a = df[["x1", "y1", "x2", "y2"]].to_numpy()
    b = df[["x2", "y2", "x1", "y1"]].to_numpy()
    tree = cKDTree(a)
    order = np.argsort(-priority, kind="stable")
    kept_by = np.full(n, -1)
    r2 = radius * np.sqrt(2) + 1e-9
    for i in order:
        if kept_by[i] >= 0:
            continue
        kept_by[i] = i
        for q in (a[i], b[i]):
            for j in tree.query_ball_point(q, r2):
                if kept_by[j] < 0:
                    d1 = np.hypot(a[j, 0] - q[0], a[j, 1] - q[1])
                    d2 = np.hypot(a[j, 2] - q[2], a[j, 3] - q[3])
                    if d1 <= radius and d2 <= radius:
                        kept_by[j] = i
    kept = np.flatnonzero(kept_by == np.arange(n))
    return kept, kept_by


def nms_priority(df: pd.DataFrame) -> np.ndarray:
    det = np.where(np.isfinite(df["detour"].to_numpy()), df["detour"].to_numpy(), 1e6)
    return det - df["euclid"].to_numpy() * 1e-6


def generate(g: WalkGraph, p: GenParams, generators=("A", "B", "C"), verbose: bool = True) -> gpd.GeoDataFrame:
    import time
    eidx = EdgeIndex(g)
    net_dist = NetworkDistance(g, cutoff=p.search_cutoff)
    fns = {"A": lambda: generator_a(eidx, p), "B": lambda: generator_b(eidx, p),
           "C": lambda: generator_c(eidx, p)}
    frames = []
    stats = []
    for gname in generators:
        t0 = time.time()
        raw = fns[gname]()
        n_raw = len(raw)
        df = filter_length(raw, p, p.stairs_max_euclid if gname == "C" else None)
        n_len = len(df)
        df = order_ends(df)
        key = df[["x1", "y1", "x2", "y2"]].round(1)
        df = df[~key.duplicated().to_numpy()]
        df = filter_sidewalk_road(df, eidx, p)
        n_sw = len(df)
        df = filter_visible(df, eidx)
        n_vis = len(df)
        df = add_network_distance(df, net_dist)
        df = filter_detour(df, p)
        stats.append({"generator": gname, "raw": n_raw, "length_ok": n_len, "not_sidewalk_road": n_sw,
                      "visible": n_vis, "detour_ok": len(df), "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"  generator {gname}: {stats[-1]}")
        frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    for gname in ("A", "B", "C"):
        allc[f"gen_{gname}"] = allc["gen"] == gname
    kept, kept_by = endpoint_nms(allc, p.dedupe_m, nms_priority(allc))
    flags = allc[["gen_A", "gen_B", "gen_C"]].groupby(kept_by).any()
    out = allc.iloc[kept].drop(columns=["gen"]).copy()
    out[["gen_A", "gen_B", "gen_C"]] = flags.loc[kept].to_numpy()
    out = out.reset_index(drop=True)
    out.insert(0, "cand_id", np.arange(len(out)))
    geom = shapely.linestrings(out[["x1", "y1", "x2", "y2"]].to_numpy().reshape(-1, 2, 2))
    gdf = gpd.GeoDataFrame(out, geometry=geom, crs=g.edges.crs)
    gdf.attrs["generator_stats"] = stats
    if verbose:
        print(f"  after dedupe: {len(gdf):,} candidates")
    return gdf
