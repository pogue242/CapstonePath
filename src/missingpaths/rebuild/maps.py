# example map figures
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def window(gdf: gpd.GeoDataFrame | None, x0, y0, x1, y1):
    if gdf is None or len(gdf) == 0:
        return gdf
    return gdf.cx[x0:x1, y0:y1]


def example_map(path: Path, focus, layers: list[tuple], title: str = "", radius: float = 120.0,
                note: str = "") -> None:
    cx, cy = focus.centroid.x, focus.centroid.y
    r = max(radius, (max(focus.bounds[2] - focus.bounds[0], focus.bounds[3] - focus.bounds[1]) / 2) + 40)
    x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    handles = []
    for gdf, kw, label in layers:
        sub = window(gdf, x0, y0, x1, y1)
        if sub is None or len(sub) == 0:
            continue
        kw = dict(kw)
        is_poly = kw.pop("kind", None) == "poly"
        sub.plot(ax=ax, **kw)
        if label:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            if is_poly:
                handles.append(Patch(facecolor=kw.get("facecolor", kw.get("color")), edgecolor="none", label=label))
            else:
                handles.append(Line2D([0], [0], color=kw.get("color", "k"), lw=kw.get("linewidth", 1.5),
                                      ls=kw.get("linestyle", "-"), label=label))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([x0 + 10, x0 + 60], [y0 + 10, y0 + 10], color="k", lw=3)
    ax.text(x0 + 35, y0 + 14, "50 m", ha="center", fontsize=8)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(title, fontsize=9)
    if note:
        fig.text(0.01, 0.01, note, fontsize=6.5, color="#444")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
