# read pedestrian areas and entrances from OSM
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pyogrio

from capstone import config
from capstone.io.osm import parse_tags

AREA_KEYS = ["highway", "area:highway", "area", "footway", "place", "surface"]
POINT_KEYS = ["entrance", "addr:housenumber", "door", "level", "barrier", "highway"]


def _clip(gdf: gpd.GeoDataFrame, boundary_4326: gpd.GeoDataFrame, city_key: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[gdf.intersects(boundary_4326.geometry.iloc[0])].copy()
    return gdf.to_crs(config.metric_crs(city_key))


def clip_ped_areas(city_key: str, boundary_4326: gpd.GeoDataFrame, force: bool = False):
    out = config.interim_dir(city_key) / "osm_ped_areas.gpkg"
    if out.exists() and not force:
        return out
    pbf = config.osm_pbf_for(city_key)
    where = ("other_tags LIKE '%\"highway\"=>%' OR other_tags LIKE '%area:highway%' "
             "OR place = 'square'")
    gdf = pyogrio.read_dataframe(pbf, layer="multipolygons", bbox=tuple(boundary_4326.total_bounds),
                                 where=where)
    if gdf.empty:
        raise RuntimeError("no pedestrian areas read")
    gdf = _clip(gdf, boundary_4326, city_key)
    tags = parse_tags(gdf["other_tags"], [k for k in AREA_KEYS if k not in gdf.columns])
    gdf = gdf.join(tags)
    tmp = out.with_suffix(".tmp.gpkg")
    gdf.to_file(tmp, layer="ped_areas", driver="GPKG")
    tmp.replace(out)
    return out


def clip_entrances(city_key: str, boundary_4326: gpd.GeoDataFrame, force: bool = False):
    out = config.interim_dir(city_key) / "osm_points.gpkg"
    if out.exists() and not force:
        return out
    pbf = config.osm_pbf_for(city_key)
    where = ("other_tags LIKE '%entrance%' OR other_tags LIKE '%addr:housenumber%' "
             "OR barrier IS NOT NULL OR highway IS NOT NULL")
    gdf = pyogrio.read_dataframe(pbf, layer="points", bbox=tuple(boundary_4326.total_bounds), where=where)
    if gdf.empty:
        raise RuntimeError("no points read")
    gdf = _clip(gdf, boundary_4326, city_key)
    tags = parse_tags(gdf["other_tags"], [k for k in POINT_KEYS if k not in gdf.columns])
    gdf = gdf.join(tags)
    tmp = out.with_suffix(".tmp.gpkg")
    gdf.to_file(tmp, layer="points", driver="GPKG")
    tmp.replace(out)
    return out


def read_ped_areas(city_key: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.interim_dir(city_key) / "osm_ped_areas.gpkg", layer="ped_areas")


def read_points(city_key: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.interim_dir(city_key) / "osm_points.gpkg", layer="points")


def walkable_areas(areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    hw = areas["highway"].fillna("") if "highway" in areas else pd.Series("", index=areas.index)
    ahw = areas["area:highway"].fillna("") if "area:highway" in areas else pd.Series("", index=areas.index)
    pl = areas["place"].fillna("") if "place" in areas else pd.Series("", index=areas.index)
    m = hw.isin(["pedestrian", "footway", "living_street"]) | ahw.isin(["pedestrian", "footway"]) | (pl == "square")
    return areas[m]
