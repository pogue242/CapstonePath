# map context around a line: water, rail, barriers, land use
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

WATERWAYS = {"river", "stream", "canal", "drain", "ditch"}
RAILWAYS = {"rail", "light_rail", "narrow_gauge", "subway", "monorail", "funicular"}
BARRIERS = {"fence", "wall", "retaining_wall", "hedge", "city_wall", "guard_rail", "handrail"}
PARK_LEISURE = {"park", "garden", "playground", "pitch", "nature_reserve", "dog_park"}
PARK_LANDUSE = {"grass", "recreation_ground", "cemetery", "forest", "meadow", "allotments", "village_green"}
LU_RES = {"residential"}
LU_IND = {"industrial", "commercial", "retail"}
TILE_M = 250.0
BIG_VERTICES = 500


def _col(df, name):
    return df[name].fillna("").to_numpy(dtype=object) if name in df else np.full(len(df), "", dtype=object)


def subdivide(polys: np.ndarray, tile: float = TILE_M, big: int = BIG_VERTICES) -> np.ndarray:
    polys = shapely.make_valid(np.asarray(polys, dtype=object))
    polys = shapely.get_parts(polys)
    polys = polys[np.isin(shapely.get_type_id(polys), [3, 6])]
    nv = shapely.get_num_coordinates(polys)
    small = list(polys[nv <= big])
    for p in polys[nv > big]:
        x0, y0, x1, y1 = p.bounds
        xs = np.arange(x0, x1 + tile, tile)
        ys = np.arange(y0, y1 + tile, tile)
        gx, gy = np.meshgrid(xs[:-1], ys[:-1])
        boxes = shapely.box(gx.ravel(), gy.ravel(), gx.ravel() + tile, gy.ravel() + tile)
        boxes = boxes[shapely.intersects(boxes, p)]
        pieces = shapely.get_parts(shapely.intersection(boxes, p))
        small.extend(pieces[np.isin(shapely.get_type_id(pieces), [3, 6]) & (shapely.area(pieces) > 0)])
    return np.asarray(small, dtype=object)


class ContextLayers:

    def __init__(self, lines: pd.DataFrame, areas: pd.DataFrame):
        lg = lines.geometry.to_numpy()
        tunnel = _col(lines, "tunnel")
        ag = areas.geometry.to_numpy()
        ww = np.isin(_col(lines, "waterway"), list(WATERWAYS)) & ~np.isin(tunnel, ["yes", "culvert"])
        water_poly = (_col(areas, "natural") == "water") | (_col(areas, "water") != "") | \
                     (_col(areas, "landuse") == "reservoir")
        rail = np.isin(_col(lines, "railway"), list(RAILWAYS)) & ~np.isin(tunnel, ["yes", "building_passage"])
        bar = np.isin(_col(lines, "barrier"), list(BARRIERS))
        park = np.isin(_col(areas, "leisure"), list(PARK_LEISURE)) | np.isin(_col(areas, "landuse"), list(PARK_LANDUSE))
        lu = _col(areas, "landuse")
        self.layers = {
            "water_lines": lg[ww], "water_polys": subdivide(ag[water_poly]),
            "rail": lg[rail], "barrier": lg[bar],
            "park": subdivide(np.asarray(shapely.get_parts(shapely.union_all(shapely.make_valid(ag[park]))),
                                         dtype=object)) if park.any() else np.zeros(0, dtype=object),
            "lu_res": subdivide(ag[np.isin(lu, list(LU_RES))]),
            "lu_ind": subdivide(ag[np.isin(lu, list(LU_IND))]),
        }
        self.trees = {k: shapely.STRtree(v) for k, v in self.layers.items() if len(v)}

    def crosses(self, geoms: np.ndarray, *names) -> np.ndarray:
        out = np.zeros(len(geoms), dtype=int)
        for n in names:
            if n in self.trees:
                si, _ = self.trees[n].query(geoms, predicate="intersects")
                out[np.unique(si)] = 1
        return out

    def frac_inside(self, geoms: np.ndarray, name: str) -> np.ndarray:
        n = len(geoms)
        if name not in self.trees:
            return np.zeros(n)
        si, pj = self.trees[name].query(geoms, predicate="intersects")
        ln = shapely.length(shapely.intersection(geoms[si], self.layers[name][pj])) if len(si) else np.zeros(0)
        tot = np.bincount(si, weights=ln, minlength=n)
        return np.minimum(tot / np.maximum(shapely.length(geoms), 1e-9), 1.0)

    def point_in(self, pts: np.ndarray, name: str) -> np.ndarray:
        out = np.zeros(len(pts), dtype=int)
        if name in self.trees:
            mi, _ = self.trees[name].query(pts, predicate="within")
            out[np.unique(mi)] = 1
        return out


def context_features(geoms: np.ndarray, layers: ContextLayers) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(geoms)))
    out["cross_water"] = layers.crosses(geoms, "water_lines", "water_polys")
    out["cross_rail"] = layers.crosses(geoms, "rail")
    out["cross_barrier"] = layers.crosses(geoms, "barrier")
    out["park_frac"] = layers.frac_inside(geoms, "park")
    mid = shapely.line_interpolate_point(geoms, 0.5, normalized=True)
    out["landuse_residential"] = layers.point_in(mid, "lu_res")
    out["landuse_industrial_commercial"] = layers.point_in(mid, "lu_ind")
    return out
