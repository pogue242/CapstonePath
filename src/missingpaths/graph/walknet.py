from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import connected_components

NOT_WALKABLE_HIGHWAY = {"motorway", "motorway_link", "trunk", "trunk_link", "construction",
                        "proposed", "abandoned", "disused", "raceway", "bus_guideway", "razed",
                        "busway", "escape", "emergency_bay", "rest_area", "services", "no"}
MOTOR_ONLY = {"motorway", "motorway_link", "trunk", "trunk_link"}
FOOT_OK = {"yes", "designated", "permissive"}
FOOT_NO = {"no", "private", "use_sidepath"}
ACCESS_NO = {"no", "private"}
EDGE_ATTRS = ["osm_id", "name", "highway", "footway", "sidewalk", "service", "tunnel", "bridge",
              "covered", "indoor", "layer", "level", "incline", "surface", "width", "foot", "access"]
ROUND_M = 0.01


def walkable_mask(lines: pd.DataFrame) -> pd.Series:
    hw = lines["highway"]
    foot = lines["foot"] if "foot" in lines else pd.Series(None, index=lines.index)
    access = lines["access"] if "access" in lines else pd.Series(None, index=lines.index)
    foot_ok = foot.isin(FOOT_OK)
    ok = hw.notna() & ~hw.isin(NOT_WALKABLE_HIGHWAY - MOTOR_ONLY)
    ok &= ~hw.isin(MOTOR_ONLY) | foot_ok
    ok &= ~foot.isin(FOOT_NO)
    ok &= ~access.isin(ACCESS_NO) | foot_ok
    ok &= (hw != "cycleway") | foot_ok
    if "area" in lines:
        ok &= lines["area"] != "yes"
    return ok


@dataclass
class WalkGraph:
    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def csr(self) -> sp.csr_matrix:
        e = self.edges
        u, v, w = e["u"].to_numpy(), e["v"].to_numpy(), e["length"].to_numpy()
        keep = u != v
        u, v, w = u[keep], v[keep], w[keep]
        df = pd.DataFrame({"a": np.minimum(u, v), "b": np.maximum(u, v), "w": w})
        df = df.groupby(["a", "b"], as_index=False)["w"].min()
        a, b, w = df["a"].to_numpy(), df["b"].to_numpy(), np.maximum(df["w"].to_numpy(), 1e-3)
        n = self.n_nodes
        return sp.csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n))

    def components(self) -> np.ndarray:
        _, lab = connected_components(self.csr(), directed=False)
        return lab


def build_graph(ways: gpd.GeoDataFrame) -> WalkGraph:
    crs = ways.crs
    parts, owner = shapely.get_parts(ways.geometry.to_numpy(), return_index=True)
    ok = shapely.get_type_id(parts) == 1
    parts, owner = parts[ok], owner[ok]
    coords, part_idx = shapely.get_coordinates(parts, return_index=True)
    if len(coords) == 0:
        raise ValueError("no line geometry to build a graph from")
    key = np.round(coords / ROUND_M).astype(np.int64)
    node_key, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    first = np.r_[True, part_idx[1:] != part_idx[:-1]]
    last = np.r_[part_idx[1:] != part_idx[:-1], True]
    is_node = first | last | (counts[inverse] > 1)

    seg_idx = np.flatnonzero(~last)
    starts = is_node & ~last
    seg_edge = (np.cumsum(starts) - 1)[seg_idx]
    ends_on_node = is_node[seg_idx + 1]
    rows_v = np.r_[seg_idx, seg_idx[ends_on_node] + 1]
    rows_e = np.r_[seg_edge, seg_edge[ends_on_node]]
    order = np.lexsort((rows_v, rows_e))
    rows_v, rows_e = rows_v[order], rows_e[order]
    geoms = shapely.linestrings(coords[rows_v], indices=rows_e)
    n_edges = len(geoms)
    first_v = np.full(n_edges, -1)
    last_v = np.full(n_edges, -1)
    chg = np.r_[True, rows_e[1:] != rows_e[:-1]]
    first_v[rows_e[chg]] = rows_v[chg]
    chg_end = np.r_[rows_e[1:] != rows_e[:-1], True]
    last_v[rows_e[chg_end]] = rows_v[chg_end]
    u = inverse[first_v]
    v = inverse[last_v]
    edge_owner = owner[part_idx[first_v]]

    used, remap = np.unique(np.r_[u, v], return_inverse=True)
    u, v = remap[: n_edges], remap[n_edges:]
    node_xy = node_key[used] * ROUND_M
    attrs = ways.iloc[edge_owner][[c for c in EDGE_ATTRS if c in ways.columns]].reset_index(drop=True)
    edges = gpd.GeoDataFrame(attrs, geometry=geoms, crs=crs)
    edges.insert(0, "edge_id", np.arange(n_edges))
    edges.insert(1, "u", u)
    edges.insert(2, "v", v)
    edges.insert(3, "length", shapely.length(geoms))
    deg = np.bincount(np.r_[u, v], minlength=len(used))
    nodes = gpd.GeoDataFrame({"node_id": np.arange(len(used)), "degree": deg},
                             geometry=shapely.points(node_xy), crs=crs)
    return WalkGraph(nodes=nodes, edges=edges)


def component_report(g: WalkGraph) -> dict:
    lab = g.components()
    sizes = np.bincount(lab)
    e_comp = lab[g.edges["u"].to_numpy()]
    km = pd.Series(g.edges["length"].to_numpy()).groupby(e_comp).sum() / 1000
    return {
        "nodes": int(g.n_nodes), "edges": int(len(g.edges)),
        "km": round(float(g.edges["length"].sum()) / 1000, 1),
        "components": int(len(sizes)),
        "largest_component_node_share": round(float(sizes.max() / sizes.sum()), 4),
        "largest_component_km": round(float(km.max()), 1),
        "dead_ends": int((g.nodes["degree"] == 1).sum()),
    }


def save_graph(g: WalkGraph, path) -> None:
    tmp = path.with_suffix(".tmp.gpkg")
    g.nodes.to_file(tmp, layer="nodes", driver="GPKG")
    g.edges.to_file(tmp, layer="edges", driver="GPKG")
    tmp.replace(path)


def load_graph(path) -> WalkGraph:
    return WalkGraph(nodes=gpd.read_file(path, layer="nodes"), edges=gpd.read_file(path, layer="edges"))
