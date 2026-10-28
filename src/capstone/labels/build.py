from __future__ import annotations

import geopandas as gpd
import pandas as pd

from capstone import config
from capstone.io import gip, osm
from capstone.labels import micro, stations

TARGET_CLASSES = ("passage", "stairs")


def station_areas_for(city_key: str) -> gpd.GeoDataFrame | None:
    path = config.interim_dir(city_key) / "osm_polygons.gpkg"
    if not path.exists():
        return None
    return osm.station_areas(osm.read_polygons(city_key, "areas"), osm.read_polygons(city_key, "buildings"))


def assign_district(gdf: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> pd.Series:
    pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=gdf.crs)
    j = gpd.sjoin(pts, districts[["district", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j["district"].reindex(gdf.index)


def build_micro_features(city_key: str, districts: gpd.GeoDataFrame | None = None,
                         require_station_areas: bool = True) -> gpd.GeoDataFrame:
    cfg = config.city(city_key)
    links = gip.read_links(city_key)
    ml = micro.micro_links(links)
    feats = micro.aggregate_features(ml)
    strata_cfg = config.locked()["strata"] if city_key == "vienna" else None
    feats["stratum"] = micro.assign_strata(feats, strata_cfg)
    areas = station_areas_for(city_key)
    if areas is None and require_station_areas:
        raise FileNotFoundError(f"OSM polygons for {city_key} not cached; station rule (c) needs them")
    subnets = cfg.get("station_filter", {}).get("subnets", [])
    feats = stations.apply_station_filter(feats, ml, links, areas, subnets)
    feats["district"] = assign_district(feats, districts) if districts is not None else None
    out = config.interim_dir(city_key) / "micro_features.gpkg"
    tmp = out.with_suffix(".tmp.gpkg")
    feats.to_file(tmp, layer="features", driver="GPKG")
    ml.to_file(tmp, layer="links", driver="GPKG")
    tmp.replace(out)
    return feats


def read_features(city_key: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.interim_dir(city_key) / "micro_features.gpkg", layer="features")


def target_features(feats: pd.DataFrame) -> pd.DataFrame:
    return feats[feats["cls"].isin(TARGET_CLASSES) & ~feats["is_station"]]


def summary_table(feats: pd.DataFrame) -> pd.DataFrame:
    t = (feats.groupby(["cls", "stratum"])
         .agg(features=("feature_id", "size"), links=("n_links", "sum"),
              km=("len_m", lambda x: round(x.sum() / 1000, 2)),
              median_len_m=("len_m", lambda x: round(x.median(), 1)),
              station=("is_station", "sum"),
              station_subnet=("station_subnet", "sum"),
              station_general=("station_general", "sum"))
         .reset_index())
    t["usable"] = t["features"] - t["station"]
    return t
