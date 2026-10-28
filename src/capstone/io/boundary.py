from __future__ import annotations

import geopandas as gpd
import pyogrio

from capstone import config


def _cache(city_key: str):
    return config.interim_dir(city_key) / "boundary.geojson"


def _from_pbf(city_key: str) -> gpd.GeoDataFrame:
    name = config.city(city_key)["place"].split(",")[0].strip()
    pbf = config.osm_pbf_for(city_key)
    safe = name.replace("'", "''")
    gdf = pyogrio.read_dataframe(pbf, layer="multipolygons",
                                 where=f"boundary = 'administrative' AND name = '{safe}'")
    if gdf.empty:
        raise RuntimeError(f"No administrative boundary named {name!r} in {pbf}")
    gdf["admin_level_num"] = gdf["admin_level"].astype(float)
    gdf = gdf.sort_values("admin_level_num").head(1)
    return gdf.set_crs("EPSG:4326", allow_override=True)[["geometry"]]


def city_boundary(city_key: str, force: bool = False) -> gpd.GeoDataFrame:
    path = _cache(city_key)
    if path.exists() and not force:
        return gpd.read_file(path).to_crs("EPSG:4326")
    try:
        import osmnx as ox
        b = ox.geocode_to_gdf(config.city(city_key)["place"])[["geometry"]]
        source = "osmnx"
    except Exception as exc:
        print(f"  OSMnx geocode failed ({exc!r}); falling back to the .pbf")
        b = _from_pbf(city_key)
        source = "pbf"
    b = b.dissolve().to_crs("EPSG:4326")
    b["source"] = source
    b.to_file(path, driver="GeoJSON")
    return b


def districts(city_key: str) -> gpd.GeoDataFrame:
    cfg = config.city(city_key).get("districts")
    if not cfg:
        raise ValueError(f"No district config for {city_key}")
    from capstone.io.osm import read_polygons
    admin = read_polygons(city_key, "admin")
    lvl = str(cfg["admin_level"])
    d = admin[admin["admin_level"].astype(str) == lvl][["name", "geometry"]].copy()
    crs = config.metric_crs(city_key)
    city_poly = city_boundary(city_key).to_crs(crs).geometry.iloc[0]
    d["inside_share"] = d.intersection(city_poly).area / d.area
    d = d[d["inside_share"] > 0.5].copy()
    d["geometry"] = d.intersection(city_poly)
    d = d.rename(columns={"name": "district"}).dissolve("district").reset_index()
    return d[["district", "geometry"]]
