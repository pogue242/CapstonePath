# pick the photo squares and make the answer key
import json
import sys

from capstone import setup_environment
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

from capstone import config
from capstone import pipeline as P
from capstone.imagery.tiles import tiles_for_bbox
from capstone.io import gip, osm
from capstone.rebuild import common as C
from capstone.rebuild import osm_extra

SQ = 160.0
ZOOM = 20
N_TRAIN_GAP, N_TRAIN_PATH, N_EVAL, N_TEST = 75, 75, 140, 50
PED_HW = {"footway", "path", "pedestrian", "steps", "corridor", "track", "bridleway", "living_street", "platform"}
CAR_HW = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
          "residential", "service", "road", "trunk", "trunk_link"}


def photo_dir(city):
    p = C.interim(city) / "photo"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(city="vienna"):
    rng = np.random.default_rng(42)
    lock = yaml.safe_load((config.CONFIG_DIR / "rebuild_locked.yaml").read_text(encoding="utf-8"))["vienna"]
    folds = lock["folds"]
    set_of = {d: "test" for d in lock["test_districts"]}
    for f in ("fold_0", "fold_1", "fold_2"):
        for d in folds[f]:
            set_of[d] = "seg_train"
    for f in ("fold_3", "fold_4"):
        for d in folds[f]:
            set_of[d] = "eval"
    dist = gpd.read_file(config.interim_dir(city) / "districts.gpkg")
    dist["set"] = dist["district"].map(set_of)
    sets = dist.dissolve("set").reset_index()
    bnd = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            s = shapely.intersection(sets.geometry.iat[i].boundary, sets.geometry.iat[j].boundary)
            if not s.is_empty:
                bnd.append(s)
    bzone = shapely.buffer(shapely.union_all(bnd), 200 + SQ / 2 * np.sqrt(2))
    gaps = gpd.read_file(C.interim(city) / "real_gaps.gpkg", layer="real_gaps")
    gaps["set"] = gaps["district"].map(set_of)
    links = gip.read_links(city)
    links = links[links["walkable"] & links["ped_only"]].reset_index(drop=True)
    links["district"] = None
    links_mid = links.geometry.interpolate(0.5, normalized=True)
    lset = gpd.sjoin(gpd.GeoDataFrame(geometry=links_mid, crs=links.crs), dist[["set", "geometry"]],
                     predicate="within", how="left")
    lset = lset[~lset.index.duplicated()]["set"].reindex(links.index)

    def pick_points(geoms, n):
        geoms = np.asarray(geoms, dtype=object)
        ok = ~shapely.intersects(shapely.centroid(geoms), bzone)
        geoms = geoms[ok]
        if len(geoms) == 0:
            return []
        idx = rng.choice(len(geoms), size=min(n, len(geoms)), replace=False)
        return [shapely.line_interpolate_point(geoms[i], 0.5, normalized=True) for i in idx]

    rows = []

    def add(points, role, gap_ids=None):
        for k, p in enumerate(points):
            rows.append({"role": role, "centre_gap": None if gap_ids is None else gap_ids[k],
                         "geometry": shapely.box(p.x - SQ / 2, p.y - SQ / 2, p.x + SQ / 2, p.y + SQ / 2)})

    tr_gaps = gaps[gaps["set"] == "seg_train"]
    add(pick_points(tr_gaps.geometry.to_numpy(), N_TRAIN_GAP), "seg_train")
    add(pick_points(links.geometry.to_numpy()[(lset == "seg_train").to_numpy()], N_TRAIN_PATH), "seg_train")
    ev = gaps[(gaps["set"] == "eval") & ~shapely.intersects(gaps.geometry.centroid.to_numpy(), bzone)]
    parts = []
    for _, sub in ev.groupby("group"):
        k = min(len(sub), max(15, int(N_EVAL * len(sub) / len(ev))))
        parts.append(sub.sample(n=k, random_state=42))
    ev = pd.concat(parts)
    add([shapely.line_interpolate_point(g, 0.5, normalized=True) for g in ev.geometry], "eval", list(ev["gap_id"]))
    te = gaps[gaps["set"] == "test"]
    te = te[~shapely.intersects(te.geometry.centroid.to_numpy(), bzone)]
    add(pick_points(te.geometry.to_numpy(), N_TEST // 2), "test")
    add(pick_points(links.geometry.to_numpy()[(lset == "test").to_numpy()], N_TEST // 2), "test")
    sq = gpd.GeoDataFrame(rows, crs=links.crs)
    sq.insert(0, "sq_id", [f"sq_{i:04d}" for i in range(len(sq))])
    pdir = photo_dir(city)
    sq.to_file(pdir / "squares.gpkg", layer="squares", driver="GPKG")

    sq_m = sq.to_crs(3857)
    tiles = {}
    for r in sq_m.itertuples():
        x0, y0, x1, y1 = r.geometry.bounds
        tiles[r.sq_id] = tiles_for_bbox(x0, y0, x1, y1, ZOOM)
    (pdir / "tiles_squares.json").write_text(json.dumps({"zoom": ZOOM, "tiles": tiles}), encoding="utf-8")
    n_tiles = len({t for v in tiles.values() for t in v})

    allsq = shapely.union_all(sq.geometry.buffer(5).to_numpy())
    ways = P.walkable_osm(city)
    hw = ways["highway"].fillna("")
    ped = ways[hw.isin(PED_HW) & ways.intersects(allsq)]
    car = ways[hw.isin(CAR_HW) & ways.intersects(allsq)]
    gl = gip.read_links(city)
    gl = gl[gl["walkable"] & gl.intersects(allsq)]
    gped = gl[gl["ped_only"]]
    gcar = gl[gl["car"]]
    pa = osm_extra.walkable_areas(osm_extra.read_ped_areas(city))
    pa = pa[pa.intersects(allsq)]
    bld = osm.read_polygons(city, "buildings")
    bld = bld[bld.intersects(allsq)]
    lines = np.r_[ped.geometry.to_numpy(), gped.geometry.to_numpy()]
    pos = shapely.union_all(np.r_[shapely.buffer(lines, 1.25), shapely.make_valid(pa.geometry.to_numpy())])
    ring = shapely.difference(shapely.union_all(shapely.buffer(lines, 2.5)), pos)
    roads = shapely.union_all(shapely.buffer(np.r_[car.geometry.to_numpy(), gcar.geometry.to_numpy()], 4.0))
    roof = shapely.union_all(shapely.make_valid(bld.geometry.to_numpy()))
    pos_open = shapely.difference(pos, roof)
    ignore = shapely.union_all([shapely.difference(roads, pos_open), ring, shapely.intersection(pos, roof)])
    out = gpd.GeoDataFrame({"kind": ["pos", "ignore", "roof"]},
                           geometry=[shapely.intersection(pos_open, allsq), shapely.intersection(ignore, allsq),
                                     shapely.intersection(roof, allsq)], crs=links.crs).to_crs(3857)
    out.to_file(pdir / "mask_vectors.gpkg", layer="mask", driver="GPKG")
    summary = {"squares": sq["role"].value_counts().to_dict(), "tiles": n_tiles,
               "eval_gaps_by_group": ev["group"].value_counts().to_dict()}
    C.write_json(C.tables() / f"{city}_photo_plan.json", summary)
    C.log("photo_plan", "tiles", n_tiles, area=city, note=json.dumps(summary["squares"]))
    print(summary)


if __name__ == "__main__":
    setup_environment()
    main(sys.argv[1] if len(sys.argv) > 1 else "vienna")
