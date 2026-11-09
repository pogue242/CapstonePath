# sort dead ends by what they touch
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from missingpaths.graph.walknet import walkable_mask

VI = ROOT / "data" / "interim" / "vienna"
TABLES = ROOT / "reports" / "rebuild" / "tables"
TOUCH_M = 0.05
WALK_AREA = (r'"(?:public_transport|highway|railway)"=>"platform"|"indoor"=>"corridor"'
             r'|"(?:area:)?highway"=>"(?:pedestrian|footway)"')


def touching(points, geoms):
    i, _ = shapely.STRtree(geoms).query(points, predicate="dwithin", distance=TOUCH_M)
    out = np.zeros(len(points), bool)
    out[np.unique(i)] = True
    return out


def read_noexit(crs):
    cache = VI / "osm_noexit.gpkg"
    if not cache.exists():
        import missingpaths
        import pyogrio
        from missingpaths import config
        bbox = tuple(gpd.read_file(VI / "districts.gpkg").to_crs(4326).total_bounds)
        pts = pyogrio.read_dataframe(config.osm_pbf_for("vienna"), layer="points", bbox=bbox,
                                     where="other_tags LIKE '%noexit%'")
        pts.to_file(cache, layer="points", driver="GPKG")
    pts = gpd.read_file(cache).to_crs(crs)
    return pts[pts.other_tags.fillna("").str.contains('"noexit"=>"yes"')]


def main():
    u = gpd.read_file(VI / "rebuild" / "unjoined.gpkg")
    u = u.merge(pd.read_csv(TABLES / "vienna_unjoined_all.csv", usecols=["gap_id", "gip_confirms"]), on="gap_id")
    pts = shapely.points(u.x1.to_numpy(), u.y1.to_numpy())
    lines = gpd.read_file(VI / "osm_lines.gpkg").to_crs(u.crs)
    hw = lines[lines.highway.notna()].reset_index(drop=True)
    non_walk = hw[~walkable_mask(hw)]
    plat_lines = lines[lines.highway.isna() & (lines.railway.isin(["platform", "platform_edge"]) | (lines.indoor == "corridor"))]
    areas = pd.concat([gpd.read_file(VI / "osm_polygons.gpkg", layer="areas").to_crs(u.crs),
                       gpd.read_file(VI / "osm_ped_areas.gpkg").to_crs(u.crs)])
    areas = areas[areas.other_tags.fillna("").str.contains(WALK_AREA)]
    noexit = read_noexit(u.crs)

    flags = {"noexit": touching(pts, noexit.geometry.to_numpy()),
             "walk_area": touching(pts, shapely.boundary(areas.geometry.to_numpy())) | touching(pts, plat_lines.geometry.to_numpy()),
             "non_walk_way": touching(pts, non_walk.geometry.to_numpy())}
    u["cls"] = np.select([flags["noexit"], flags["walk_area"], flags["non_walk_way"]],
                         ["marked_dead_end", "via_platform_or_area", "via_non_walk_way"], "gap_in_drawing")
    for k, v in flags.items():
        u[k] = v
    cols = ["gap_id", "district", "gap_m", "netdist", "way_kind", "target_kind", "osm_id_end", "osm_id_target",
            "gip_confirms", "cls", *flags]
    u[cols].to_csv(TABLES / "vienna_unjoined_classes.csv", index=False)
    conf = u.gip_confirms.astype(str).str.lower() == "true"
    summary = {"total": len(u), "by_class": u.cls.value_counts().to_dict(),
               "gip_confirms_by_class": u[conf].cls.value_counts().to_dict()}
    (TABLES / "vienna_unjoined_classes.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
