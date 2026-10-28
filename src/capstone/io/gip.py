from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from capstone import config

BIT_PED, BIT_BIKE, BIT_CAR, BIT_BUS = 1, 2, 4, 8

LINK_COLUMNS = ["object_id", "short_id", "node_from_short_id", "node_to_short_id", "edge_short_id",
                "name_text_hr", "access_tow", "access_bkw", "length", "functional_class",
                "form_of_way", "level_intermediate", "subnet", "edge_category", "regional_code",
                "oneway_pedestrian", "urban", "construction_state"]


def find_layer(gpkg: Path, wanted: str) -> str | None:
    names = [n for n, _ in pyogrio.list_layers(gpkg)]
    for n in names:
        if n.lower() == wanted.lower():
            return n
    return None


def read_clipped(path: Path, layer: str, boundary_4326: gpd.GeoDataFrame,
                 columns: list[str] | None = None) -> gpd.GeoDataFrame:
    info = pyogrio.read_info(path, layer=layer)
    crs = info["crs"] or "EPSG:4326"
    b = boundary_4326.to_crs(crs)
    if columns is not None:
        have = {f.lower(): f for f in info["fields"]}
        columns = [have[c] for c in columns if c in have]
    gdf = pyogrio.read_dataframe(path, layer=layer, bbox=tuple(b.total_bounds), columns=columns)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    gdf = gdf.rename(columns={c: c.lower() for c in gdf.columns if c != "geometry"})
    return gdf[gdf.intersects(b.geometry.iloc[0])].copy()


def add_access_flags(links: pd.DataFrame) -> pd.DataFrame:
    for c in ("access_tow", "access_bkw"):
        links[c] = pd.to_numeric(links[c], errors="coerce").fillna(0).astype("int64")
    any_access = links["access_tow"] | links["access_bkw"]
    links["walkable"] = (any_access & BIT_PED) > 0
    links["car"] = (any_access & BIT_CAR) > 0
    links["ped_only"] = links["walkable"] & ~links["car"]
    for c in ("form_of_way", "subnet", "level_intermediate", "functional_class", "edge_category"):
        if c in links:
            links[c] = pd.to_numeric(links[c], errors="coerce")
    return links


def clip_gip(city_key: str, boundary_4326: gpd.GeoDataFrame, force: bool = False) -> Path:
    out = config.interim_dir(city_key) / "gip.gpkg"
    if out.exists() and not force:
        return out
    crs = config.metric_crs(city_key)
    net = config.data_path("gip_network")
    tmp = out.with_suffix(".tmp.gpkg")
    links = read_clipped(net, find_layer(net, "link"), boundary_4326, LINK_COLUMNS)
    links = add_access_flags(links).to_crs(crs)
    links["len_m"] = links.length
    links.to_file(tmp, layer="link", driver="GPKG")
    nodes = read_clipped(net, find_layer(net, "node"), boundary_4326,
                         ["short_id", "level", "form_of_node", "edge_degree"])
    nodes.to_crs(crs).to_file(tmp, layer="node", driver="GPKG")
    lu = read_clipped(net, find_layer(net, "linear_use"), boundary_4326,
                      ["short_id", "edge_short_id", "base_type", "width_average", "length_m"])
    lu = lu.to_crs(crs)
    lu["len_m"] = lu.length
    lu.to_file(tmp, layer="linear_use", driver="GPKG")
    try:
        ref = config.data_path("gip_reference")
        br = read_clipped(ref, find_layer(ref, "brunnel"), boundary_4326)
        br.to_crs(crs).to_file(tmp, layer="brunnel", driver="GPKG")
    except FileNotFoundError as exc:
        print(f"  brunnel skipped: {exc}")
    tmp.replace(out)
    return out


def read_links(city_key: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.interim_dir(city_key) / "gip.gpkg", layer="link")


def summary_counts(links: pd.DataFrame) -> dict:
    walk = links[links["walkable"]]
    po = links[links["ped_only"]]
    lvl = pd.to_numeric(walk.get("level_intermediate"), errors="coerce").fillna(0)
    return {
        "links_total": int(len(links)),
        "links_km": round(float(links["len_m"].sum()) / 1000, 1),
        "walkable_links": int(len(walk)),
        "walkable_km": round(float(walk["len_m"].sum()) / 1000, 1),
        "ped_only_links": int(len(po)),
        "ped_only_km": round(float(po["len_m"].sum()) / 1000, 1),
        "walkable_nonzero_level": int((lvl != 0).sum()),
    }
