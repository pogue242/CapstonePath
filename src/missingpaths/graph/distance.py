# walking distance on the graph
from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse.csgraph import dijkstra

from missingpaths.graph.walknet import WalkGraph


class NetworkDistance:
    def __init__(self, g: WalkGraph, cutoff: float, tile_m: float = 250.0):
        self.cutoff = float(cutoff)
        self.tile = float(tile_m)
        self.csr = g.csr()
        xy = shapely.get_coordinates(g.nodes.geometry.to_numpy())
        self.x, self.y = xy[:, 0], xy[:, 1]
        e = g.edges
        self.eu = e["u"].to_numpy()
        self.ev = e["v"].to_numpy()
        self.elen = e["length"].to_numpy()

    def node_dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        out = np.full(len(a), np.inf)
        if len(a) == 0:
            return out
        same = a == b
        out[same] = 0.0
        idx = np.flatnonzero(~same)
        if len(idx) == 0:
            return out
        tx = np.floor(self.x[a[idx]] / self.tile).astype(np.int64)
        ty = np.floor(self.y[a[idx]] / self.tile).astype(np.int64)
        order = np.lexsort((ty, tx))
        idx, tx, ty = idx[order], tx[order], ty[order]
        brk = np.flatnonzero(np.r_[True, (tx[1:] != tx[:-1]) | (ty[1:] != ty[:-1]), True])
        local = np.full(len(self.x), -1, dtype=np.int64)
        for s, e in zip(brk[:-1], brk[1:]):
            grp = idx[s:e]
            src = np.unique(a[grp])
            x0, x1 = self.x[src].min() - self.cutoff, self.x[src].max() + self.cutoff
            y0, y1 = self.y[src].min() - self.cutoff, self.y[src].max() + self.cutoff
            sub = np.flatnonzero((self.x >= x0) & (self.x <= x1) & (self.y >= y0) & (self.y <= y1))
            local[sub] = np.arange(len(sub))
            m = self.csr[sub][:, sub]
            dmat = dijkstra(m, directed=False, indices=local[src], limit=self.cutoff)
            rows = np.searchsorted(src, a[grp])
            cols = local[b[grp]]
            ok = cols >= 0
            vals = np.full(len(grp), np.inf)
            vals[ok] = dmat[rows[ok], cols[ok]]
            out[grp] = vals
            local[sub] = -1
        out[out > self.cutoff] = np.inf
        return out

    def point_dist(self, e1, t1, e2, t2) -> np.ndarray:
        e1, e2 = np.asarray(e1), np.asarray(e2)
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        u1, v1, l1 = self.eu[e1], self.ev[e1], self.elen[e1]
        u2, v2, l2 = self.eu[e2], self.ev[e2], self.elen[e2]
        n = len(e1)
        a = np.r_[u1, u1, v1, v1]
        b = np.r_[u2, v2, u2, v2]
        off = np.r_[t1 + t2, t1 + (l2 - t2), (l1 - t1) + t2, (l1 - t1) + (l2 - t2)]
        d = self.node_dist(a, b) + off
        best = d.reshape(4, n).min(axis=0)
        same = e1 == e2
        best[same] = np.minimum(best[same], np.abs(t1[same] - t2[same]))
        best[best > self.cutoff] = np.inf
        return best
