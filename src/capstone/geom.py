from __future__ import annotations

import numpy as np
import pandas as pd
import shapely


def bearing_at(lines: np.ndarray, dists: np.ndarray, h: float = 1.0) -> np.ndarray:
    lens = shapely.length(lines)
    a = shapely.line_interpolate_point(lines, np.clip(dists - h, 0, None))
    b = shapely.line_interpolate_point(lines, np.minimum(dists + h, lens))
    dx = shapely.get_x(b) - shapely.get_x(a)
    dy = shapely.get_y(b) - shapely.get_y(a)
    return np.degrees(np.arctan2(dy, dx)) % 180


def angle_diff(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    d = np.abs(np.asarray(b1) - np.asarray(b2)) % 180
    return np.minimum(d, 180 - d)


def segment_bearing(x1, y1, x2, y2) -> np.ndarray:
    return np.degrees(np.arctan2(np.asarray(y2) - y1, np.asarray(x2) - x1)) % 180


def sample_along(lines: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lens = shapely.length(lines)
    n_pts = np.maximum(np.ceil(lens / step).astype(int), 2)
    owner = np.repeat(np.arange(len(lines)), n_pts)
    frac = np.concatenate([np.linspace(0.02, 0.98, k) for k in n_pts]) if len(lines) else np.array([])
    d = frac * lens[owner]
    pts = shapely.line_interpolate_point(lines[owner], d)
    return owner, d, pts


def aligned_coverage(src: np.ndarray, ref: np.ndarray, tol: float, step: float, max_angle: float,
                     ref_tree: shapely.STRtree | None = None, ref_owner: np.ndarray | None = None):
    src = np.asarray(src, dtype=object)
    ref = np.asarray(ref, dtype=object)
    if len(src) == 0:
        return np.zeros(0), pd.DataFrame(columns=["src", "ref", "n_points"])
    owner, d, pts = sample_along(src, step)
    b_src = bearing_at(src[owner], d)
    tree = ref_tree if ref_tree is not None else shapely.STRtree(ref)
    if len(ref) == 0:
        pi = ri = np.zeros(0, dtype=int)
    else:
        pi, ri = tree.query(pts, predicate="dwithin", distance=tol)
    if len(pi):
        rg = ref[ri]
        rd = shapely.line_locate_point(rg, pts[pi])
        ok = angle_diff(b_src[pi], bearing_at(rg, rd)) <= max_angle
        pi, ri = pi[ok], ri[ok]
    covered = np.zeros(len(pts), dtype=bool)
    covered[pi] = True
    n_per_src = np.bincount(owner, minlength=len(src))
    cov = np.bincount(owner, weights=covered, minlength=len(src)) / np.maximum(n_per_src, 1)
    if ref_owner is not None:
        ri = np.asarray(ref_owner)[ri]
    hits = (pd.DataFrame({"src": owner[pi], "ref": ri, "pt": pi})
            .drop_duplicates(["pt", "ref"])
            .groupby(["src", "ref"]).size().rename("n_points").reset_index())
    return cov, hits


def match_status(cov: np.ndarray, matched_min: float, partial_min: float,
                 names=("matched", "partial", "gip_only")) -> np.ndarray:
    return np.where(cov >= matched_min, names[0], np.where(cov >= partial_min, names[1], names[2]))


def explode_lines(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts, idx = shapely.get_parts(np.asarray(geoms, dtype=object), return_index=True)
    keep = shapely.length(parts) > 0
    return parts[keep], idx[keep]
