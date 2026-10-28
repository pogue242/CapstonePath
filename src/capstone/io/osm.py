from __future__ import annotations

import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from capstone import config

HSTORE = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')

LINE_KEYS = ["footway", "foot", "access", "tunnel", "bridge", "covered", "indoor", "layer",
             "level", "incline", "sidewalk", "area", "surface", "width", "service",
             "public_transport", "motor_vehicle", "vehicle", "construction"]
POLY_KEYS = ["railway", "public_transport", "water", "layer", "location", "building:levels",
             "tunnel", "indoor", "area:highway", "highway"]

PED_HIGHWAYS = {"footway", "path", "pedestrian", "steps", "living_street",
                "corridor", "track", "bridleway"}


def parse_tags(other_tags: pd.Series, keys: list[str]) -> pd.DataFrame:
    tags = other_tags.map(lambda s: dict(HSTORE.findall(s)) if isinstance(s, str) else {})
    return pd.DataFrame({k: tags.map(lambda t, k=k: t.get(k)) for k in keys}, index=other_tags.index)


def _read_bbox(pbf: Path, layer: str, boundary_4326: gpd.GeoDataFrame, where: str | None,
               columns: list[str] | None = None) -> gpd.GeoDataFrame:
    t0 = time.time()
    gdf = pyogrio.read_dataframe(pbf, layer=layer, bbox=tuple(boundary_4326.total_bounds),
                                 where=where, columns=columns)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    print(f"  read {layer}: {len(gdf):,} features in bbox ({time.time() - t0:,.0f} s)")
    if gdf.empty:
        raise RuntimeError(f"0 features read from {pbf.name}:{layer}; refusing to cache an empty clip")
    return gdf


def pedestrian_mask(gdf: pd.DataFrame) -> pd.Series:
    foot_ok = gdf["foot"].isin(["yes", "designated", "permissive"])
    foot_no = gdf["foot"].isin(["no", "private"]) | (gdf["access"].isin(["no", "private"]) & ~foot_ok)
    return (gdf["highway"].isin(PED_HIGHWAYS) | foot_ok) & ~foot_no


def clip_osm_lines(city_key: str, boundary_4326: gpd.GeoDataFrame, force: bool = False) -> Path:
    out = config.interim_dir(city_key) / "osm_lines.gpkg"
    if out.exists() and not force:
        return out
    pbf = config.osm_pbf_for(city_key)
    where = ("highway IS NOT NULL OR railway IS NOT NULL OR waterway IS NOT NULL "
             "OR barrier IS NOT NULL")
    gdf = _read_bbox(pbf, "lines", boundary_4326, where)
    poly = boundary_4326.geometry.iloc[0]
    gdf = gdf[gdf.intersects(poly)].copy()
    gdf = gdf.join(parse_tags(gdf["other_tags"], [k for k in LINE_KEYS if k not in gdf.columns]))
    gdf = gdf.to_crs(config.metric_crs(city_key))
    gdf["len_m"] = gdf.length
    tmp = out.with_suffix(".tmp.gpkg")
    gdf.to_file(tmp, layer="lines", driver="GPKG")
    tmp.replace(out)
    return out


def clip_osm_polygons(city_key: str, boundary_4326: gpd.GeoDataFrame, force: bool = False) -> Path:
    out = config.interim_dir(city_key) / "osm_polygons.gpkg"
    if out.exists() and not force:
        return out
    pbf = config.osm_pbf_for(city_key)
    where = ("building IS NOT NULL OR landuse IS NOT NULL OR leisure IS NOT NULL "
             "OR natural IS NOT NULL OR amenity IS NOT NULL OR boundary IS NOT NULL "
             "OR other_tags LIKE '%railway%' OR other_tags LIKE '%public_transport%' "
             "OR other_tags LIKE '%water%' OR man_made IS NOT NULL")
    gdf = _read_bbox(pbf, "multipolygons", boundary_4326, where)
    poly = boundary_4326.geometry.iloc[0]
    gdf = gdf[gdf.intersects(poly)].copy()
    gdf = gdf.join(parse_tags(gdf["other_tags"], [k for k in POLY_KEYS if k not in gdf.columns]))
    gdf = gdf.to_crs(config.metric_crs(city_key))
    tmp = out.with_suffix(".tmp.gpkg")
    is_admin = gdf["boundary"].eq("administrative") & gdf["admin_level"].notna()
    is_building = gdf["building"].notna() & ~is_admin
    layers = {
        "buildings": gdf[is_building],
        "admin": gdf[is_admin],
        "areas": gdf[~is_building & ~is_admin],
    }
    for name, part in layers.items():
        part.to_file(tmp, layer=name, driver="GPKG")
    tmp.replace(out)
    return out


def read_lines(city_key: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.interim_dir(city_key) / "osm_lines.gpkg", layer="lines")


def read_polygons(city_key: str, layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.interim_dir(city_key) / "osm_polygons.gpkg", layer=layer)


def station_areas(areas: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    parts = []
    m = pd.Series(False, index=areas.index)
    for col, vals in (("railway", ["station"]), ("public_transport", ["station"]),
                      ("building", ["train_station"])):
        if col in areas:
            m |= areas[col].isin(vals)
    parts.append(areas.loc[m, ["geometry"]])
    if buildings is not None and len(buildings):
        mb = buildings["building"].eq("train_station")
        for col in ("railway", "public_transport"):
            if col in buildings:
                mb |= buildings[col].eq("station")
        parts.append(buildings.loc[mb, ["geometry"]])
    out = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=areas.crs)


def summary_counts(lines: gpd.GeoDataFrame) -> dict:
    hw = lines[lines["highway"].notna()]
    ped = hw[pedestrian_mask(hw)]
    return {
        "ped_ways": int(len(ped)),
        "ped_km": round(float(ped["len_m"].sum()) / 1000, 1),
        "steps": int((ped["highway"] == "steps").sum()),
        "tunnel=building_passage": int((ped["tunnel"] == "building_passage").sum()),
        "footway": int((ped["highway"] == "footway").sum()),
        "corridor": int((ped["highway"] == "corridor").sum()),
    }
