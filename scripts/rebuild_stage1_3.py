# graph, what OSM is missing, and joining errors
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from missingpaths import setup_environment
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from missingpaths import config
from missingpaths import pipeline as P
from missingpaths.graph import walknet
from missingpaths.io import gip, osm
from missingpaths.labels import build as labels_build
from missingpaths.labels import micro
from missingpaths.rebuild import common as C
from missingpaths.rebuild import inventory as INV
from missingpaths.rebuild import maps, osm_extra
from missingpaths.rebuild import unjoined as UJ

P.tables_dir = lambda: C.tables()
SEED = 42


START = time.time()


def t(msg):
    print(f"[{time.time() - START:7.0f}s] {msg}", flush=True)


def repro(city):
    res = P.reproduction_check(city)
    t(f"reproduction: all_match={res['all_match']}")
    m2 = P.m2_labels(city)
    t(f"m2: {m2['usable_targets']}")
    m3 = P.m3_match(city, write_review=False)
    t(f"m3: {m3['status_targets']}")
    links = gip.read_links(city)
    ml = micro.micro_links(links)
    feats = labels_build.read_features(city)
    small = feats[feats["cls"].isin(["passage", "stairs"])]
    chain = {
        "links_total": int(len(links)),
        "walkable_links": int(links["walkable"].sum()),
        "small_link_segments": int(ml["cls"].isin(["passage", "stairs"]).sum()),
        "small_link_features": int(len(small)),
        "features_passage": int((small["cls"] == "passage").sum()),
        "features_stairs": int((small["cls"] == "stairs").sum()),
        "after_station_filter": int((~small["is_station"]).sum()),
        "match_status_targets": m3["status_targets"],
        "expected": {"links_total": 161395, "walkable_links": 151670, "small_link_segments": 8619,
                     "small_link_features": 3400, "after_station_filter": 2074,
                     "match": {"matched": 1465, "partial": 173, "gip_only": 436}},
        "first_study_check": res,
    }
    ok = (chain["links_total"] == 161395 and chain["walkable_links"] == 151670
          and chain["small_link_segments"] == 8619 and chain["small_link_features"] == 3400
          and chain["after_station_filter"] == 2074
          and m3["status_targets"] == {"matched": 1465, "partial": 173, "gip_only": 436})
    chain["all_reproduced"] = bool(ok)
    C.write_json(C.tables() / f"{city}_reproduction_chain.json", chain)
    C.log("repro", "all_reproduced", int(ok), area=city,
          note=";".join(f"{k}={v}" for k, v in chain.items() if isinstance(v, int)))
    t(f"chain: {chain}")


def graph(city):
    ways = P.walkable_osm(city)
    g = walknet.build_graph(ways)
    walknet.save_graph(g, P.graph_path(city, "original"))
    rep = walknet.component_report(g)
    C.write_json(C.tables() / f"{city}_graph_original.json", rep)
    C.log("graph", "components", rep["components"], area=city, note=str(rep))
    t(f"graph: {rep}")


def districts_or_city(city):
    p = config.interim_dir(city) / "districts.gpkg"
    if p.exists():
        return gpd.read_file(p)
    from missingpaths.io import boundary
    b = boundary.city_boundary(city).to_crs(config.metric_crs(city))
    return gpd.GeoDataFrame({"district": [city]}, geometry=b.geometry.to_numpy(), crs=b.crs)


def barriers(city):
    lines = osm.read_lines(city)
    b = lines["barrier"].fillna("") if "barrier" in lines else pd.Series("", index=lines.index)
    return lines.loc[b.isin(UJ.BARRIERS)].geometry.to_numpy()


def unjoined(city):
    g = walknet.load_graph(P.graph_path(city, "original"))
    gaps, counts = UJ.find_unjoined(g, barriers(city))
    gdf = UJ.gaps_gdf(gaps, g.edges.crs)
    d = districts_or_city(city)
    gdf["district"] = labels_build.assign_district(gdf, d) if len(gdf) else None
    gdf.insert(0, "gap_id", [f"uj_{i}" for i in range(len(gdf))])
    out = C.interim(city) / "unjoined.gpkg"
    gdf.to_file(out, layer="unjoined", driver="GPKG")
    C.write_json(C.tables() / f"{city}_unjoined_counts.json", counts)
    C.log("unjoined", "unjoined_gaps", counts["unjoined_gaps"], area=city, note=str(counts))
    t(f"unjoined: {counts}")


def read_unjoined(city):
    return gpd.read_file(C.interim(city) / "unjoined.gpkg", layer="unjoined")


def inventory(city):
    lk = config.locked()
    m = lk["match"]
    step, ang = m["step_m"], m["angle_deg"]
    links = gip.read_links(city)
    links = links[links["walkable"]].explode(index_parts=False).reset_index(drop=True)
    links = links[links.geometry.notna() & ~links.geometry.is_empty].reset_index(drop=True)
    links["len_m"] = links.length
    subnets = config.city(city).get("station_filter", {}).get("subnets", [])
    strata = lk["strata"] if city == "vienna" else {"passage_historic_subnets": [], "passage_estate_subnets": []}
    links = links.join(INV.link_types(links, strata, subnets))
    sa = osm.station_areas(osm.read_polygons(city, "areas"), osm.read_polygons(city, "buildings"))
    geoms = links.geometry.to_numpy()
    links["station_area_share"] = INV.share_inside(geoms, sa.geometry.to_numpy())
    links["is_station"] = (links["st_level"] | links["st_subnet"] | links["st_indoor_fow"]
                           | (links["station_area_share"] >= INV.STATION_AREA_MIN_SHARE))
    t(f"links {len(links):,}; stations {int(links['is_station'].sum()):,}")

    ways = P.walkable_osm(city)
    ref = ways.geometry.to_numpy()
    tree = shapely.STRtree(ref)
    for tol in (5, 8, 12):
        links[f"cov{tol}"] = INV.coverage(geoms, ref, tol, ang, step, tree=tree).round(3)
        t(f"coverage {tol} m done")
    links["status"] = INV.status_of(links["cov8"].to_numpy())

    off = (links["status"] != "in_osm").to_numpy()
    roads = ways[ways["highway"].isin(INV.ROAD_HIGHWAYS)].reset_index(drop=True)
    cand = off & (links["ltype"] == "footpath").to_numpy()
    links["road_cov15"] = 0.0
    links.loc[cand, "road_cov15"] = INV.coverage(geoms[cand], roads.geometry.to_numpy(), INV.SIDEWALK_TOL_M,
                                                 ang, step).round(3)
    links["sidewalk_like"] = links["road_cov15"] >= 0.8
    sw = links["sidewalk_like"].to_numpy()
    if sw.any():
        rtree = shapely.STRtree(roads.geometry.to_numpy())
        mid = shapely.line_interpolate_point(geoms[sw], 0.5, normalized=True)
        near = rtree.query_nearest(mid, return_distance=False, all_matches=False)[1]
        tag = roads["sidewalk"].fillna("").to_numpy(dtype=object)[near] if "sidewalk" in roads else ""
        links.loc[sw, "osm_road_sidewalk_tag"] = tag
    pa = osm_extra.walkable_areas(osm_extra.read_ped_areas(city))
    links["plaza_share"] = 0.0
    links.loc[cand, "plaza_share"] = INV.share_inside(geoms[cand], pa.geometry.to_numpy()).round(3)
    links["plaza_like"] = links["plaza_share"] >= INV.PLAZA_MIN_SHARE
    t(f"sidewalk-like {int(links['sidewalk_like'].sum()):,}; plaza-like {int(links['plaza_like'].sum()):,}")
    gaps = read_unjoined(city)
    ref2 = np.r_[ref, gaps.geometry.to_numpy()]
    links["cov8_join"] = links["cov8"]
    links.loc[off, "cov8_join"] = INV.coverage(geoms[off], ref2, 8, ang, step).round(3)
    links["unjoined_like"] = (links["cov8_join"] >= 0.8) & off
    areas = osm.read_polygons(city, "areas")
    for name, spec in (("cemetery_path", INV.CEMETERY), ("rural_track", INV.RURAL)):
        sh = np.zeros(len(links))
        sh[cand] = INV.share_inside(geoms[cand], INV.landuse_polys(areas, spec))
        links[f"{name}_share"] = sh.round(3)
        links[f"{name}_like"] = sh >= INV.LANDUSE_MIN_SHARE
    links["inv_class"] = INV.classify(links)
    d = districts_or_city(city)
    links["district"] = labels_build.assign_district(links, d)
    t("classified")

    gref = links.geometry.to_numpy()
    ways["gip_cov8"] = INV.coverage(ways.geometry.to_numpy(), gref, 8, ang, step).round(3)
    ways["osm_only"] = ways["gip_cov8"] < 0.3
    t(f"osm-only ways {int(ways['osm_only'].sum()):,}")

    foot = links[(links["inv_class"] == "missing") & (links["group"] == "footpath")]
    gfoot = INV.merge_gap_links(foot) if len(foot) else None
    mf = gpd.read_file(config.interim_dir(city) / "match.gpkg", layer="gip_features")
    mg = mf[(mf["status"] == "gip_only") & mf["target"].astype(bool)].reset_index(drop=True)
    mg["cov8_join"] = INV.coverage(mg.geometry.to_numpy(), ref2, 8, ang, step).round(3)
    smap = {"passage_estate": "estate_passage", "passage_historic": "historic_passage",
            "passage_other": "other", "stairs": "stairs", "passage": "passage_pooled"}
    micro_gaps = gpd.GeoDataFrame({
        "group": mg["stratum"].map(smap), "ltype": mg["stratum"], "n_links": mg["n_links"],
        "len_m": mg["len_m"], "cov_mean": mg["coverage"], "link_ids": mg["feature_id"],
        "unjoined_like": mg["cov8_join"] >= 0.8}, geometry=mg.geometry, crs=mg.crs)
    chain_micro = {"micro_gip_only_targets": int(len(mg)),
                   "micro_unjoined_like": int(micro_gaps["unjoined_like"].sum())}
    micro_gaps = micro_gaps[~micro_gaps["unjoined_like"]].drop(columns="unjoined_like")
    gaps_all = pd.concat([gfoot, micro_gaps], ignore_index=True) if gfoot is not None else micro_gaps
    gaps_all = gpd.GeoDataFrame(gaps_all, geometry="geometry", crs=links.crs)
    n_all = int(len(gaps_all))
    short = gaps_all["len_m"] < INV.MIN_GAP_LEN_M
    gaps_all = gaps_all[~short].reset_index(drop=True)
    gaps_all["district"] = labels_build.assign_district(gaps_all, d)
    gaps_all.insert(0, "gap_id", [f"gap_{i}" for i in range(len(gaps_all))])
    gaps_all.to_file(C.interim(city) / "real_gaps.gpkg", layer="real_gaps", driver="GPKG")

    tdir = C.tables()
    km = lambda s: round(float(s.sum()) / 1000, 1)
    rows = []

    def add(step_name, mask, level="link"):
        sub = links[mask]
        rows.append({"step": step_name, "level": level, "links": int(mask.sum()), "km": km(sub["len_m"])})
    allm = np.ones(len(links), dtype=bool)
    add("0 all walkable GIP links", allm)
    add("1 not fully in OSM (coverage < 0.8)", off)
    add("2 missing (coverage < 0.3)", (links["status"] == "missing").to_numpy())
    c = links["inv_class"].to_numpy()
    missing = (links["status"] == "missing").to_numpy()
    m_ = missing.copy()
    for name, cls in (("3 minus stations / indoor", "station_or_indoor"),
                      ("4 minus sidewalk drawn differently", "sidewalk_drawn_differently"),
                      ("5 minus plaza drawn as an area", "plaza_drawn_as_area"),
                      ("6 minus joining errors (unjoined)", "unjoined_in_osm")):
        m_ = m_ & (c != cls)
        add(name, m_)
    m_ = m_ & (links["group"] != "street").to_numpy()
    add("7 minus car-accessible streets", m_)
    m_ = m_ & (c != "cemetery_path")
    add("8 minus footpaths inside cemeteries (grave-field rows)", m_)
    m_ = m_ & (c != "rural_track")
    add("9 minus footpaths in fields / vineyards / forest (rural tracks)", m_)
    chain = pd.DataFrame(rows)
    chain.to_csv(tdir / f"{city}_inventory_filter_chain.csv", index=False)

    by_class = (links.groupby(["group", "inv_class"]).agg(links=("len_m", "size"), km=("len_m", lambda s: round(s.sum() / 1000, 2)))
                .reset_index())
    by_class.to_csv(tdir / f"{city}_inventory_by_group_class.csv", index=False)
    gap_tab = (gaps_all.groupby("group").agg(gaps=("gap_id", "size"), km=("len_m", lambda s: round(s.sum() / 1000, 2)),
                                             median_len_m=("len_m", lambda s: round(s.median(), 1))).reset_index())
    gap_tab.to_csv(tdir / f"{city}_real_gaps_by_group.csv", index=False)
    pd.crosstab(gaps_all["district"], gaps_all["group"], margins=True).to_csv(tdir / f"{city}_real_gaps_by_district.csv")
    osm_only = (ways[ways["osm_only"]].groupby("highway").agg(ways=("len_m", "size"), km=("len_m", lambda s: round(s.sum() / 1000, 2)))
                .sort_values("ways", ascending=False).reset_index())
    osm_only.to_csv(tdir / f"{city}_osm_only_by_highway.csv", index=False)
    sens = []
    ns = links[~links["is_station"]]
    for tol in (5, 8, 12):
        for cut in (0.7, 0.8, 0.9):
            stt = INV.status_of(ns[f"cov{tol}"].to_numpy(), in_min=cut)
            for grp in sorted(ns["group"].unique()):
                gm = (ns["group"] == grp).to_numpy()
                sens.append({"tol_m": tol, "in_osm_cutoff": cut, "group": grp, "links": int(gm.sum()),
                             "in_osm": int((stt[gm] == "in_osm").sum()), "partly": int((stt[gm] == "partly").sum()),
                             "missing": int((stt[gm] == "missing").sum()),
                             "missing_share": round(float((stt[gm] == "missing").mean()), 4)})
    pd.DataFrame(sens).to_csv(tdir / f"{city}_inventory_sensitivity.csv", index=False)
    summary = {"chain": rows, "micro_chain": chain_micro, "gaps_before_min_length": n_all,
               "gaps_dropped_short": int(short.sum()),
               "real_gaps_by_group": gap_tab.set_index("group")["gaps"].to_dict(),
               "osm_only_ways": int(ways["osm_only"].sum()), "osm_only_km": km(ways.loc[ways["osm_only"], "len_m"]),
               "sidewalk_like_with_osm_sidewalk_tag": (links.loc[links["sidewalk_like"], "osm_road_sidewalk_tag"]
                                                       .fillna("").ne("").sum() if "osm_road_sidewalk_tag" in links else 0)}
    C.write_json(tdir / f"{city}_inventory_summary.json", summary)
    for grp, n in summary["real_gaps_by_group"].items():
        C.log("inventory", f"real_gaps_{grp}", n, area=city)
    t(f"summary: {summary['real_gaps_by_group']}")

    out = config.reports_dir("inventory") / f"{city}_inventory.gpkg"
    tmp = out.with_suffix(".tmp.gpkg")
    keep = ["object_id", "ltype", "group", "district", "len_m", "cov5", "cov8", "cov12", "status", "inv_class",
            "road_cov15", "plaza_share", "cov8_join", "cemetery_path_share", "rural_track_share", "is_station",
            "geometry"]
    links[[c for c in keep if c in links]].to_file(tmp, layer="gip_links_by_status", driver="GPKG")
    gaps_all.to_file(tmp, layer="real_gaps", driver="GPKG")
    wk = ["osm_id", "name", "highway", "footway", "tunnel", "gip_cov8", "osm_only", "len_m", "geometry"]
    ways[[c for c in wk if c in ways]].to_file(tmp, layer="osm_walk_ways", driver="GPKG")
    gaps.to_file(tmp, layer="unjoined_gaps", driver="GPKG")
    d.to_file(tmp, layer="districts", driver="GPKG")
    tmp.replace(out)
    from missingpaths.rebuild import qgis
    qgis.write_project(out.parent / f"{city}_inventory.qgs", out.name, [
        {"name": "real_gaps", "geom": "Line", "color": "215,48,31,255", "width": 1.4, "cat_attr": "group",
         "cats": {"footpath": "227,26,28,255", "estate_passage": "255,127,0,255", "historic_passage": "106,61,154,255",
                  "stairs": "31,120,180,255", "other": "177,89,40,255"}},
        {"name": "unjoined_gaps", "geom": "Line", "color": "255,0,255,255", "width": 1.4},
        {"name": "gip_links_by_status", "geom": "Line", "color": "150,150,150,255", "width": 0.6, "cat_attr": "inv_class",
         "cats": {"in_osm": "166,206,227,255", "partly": "253,191,111,255", "missing": "227,26,28,255",
                  "sidewalk_drawn_differently": "178,223,138,255", "plaza_drawn_as_area": "51,160,44,255",
                  "unjoined_in_osm": "255,0,255,255", "station_or_indoor": "120,120,120,255",
                  "cemetery_path": "202,178,214,255", "rural_track": "177,89,40,255"}},
        {"name": "osm_walk_ways", "geom": "Line", "color": "40,40,40,255", "width": 0.4},
        {"name": "districts", "geom": "Polygon", "color": "0,0,0,40", "visible": False},
    ])
    t(f"wrote {out} and QGIS project")
    links.drop(columns="geometry").to_parquet(C.interim(city) / "inventory_links.parquet")
    example_maps(city, gaps_all, links, ways)


def example_maps(city, gaps_all, links, ways):
    bld = osm.read_polygons(city, "buildings")
    rng = np.random.default_rng(SEED)
    fdir = C.figures() / "inventory"
    gl = links[["inv_class", "geometry"]]
    for grp in INV.GAP_GROUPS:
        sub = gaps_all[gaps_all["group"] == grp]
        if sub.empty:
            continue
        pick = rng.choice(len(sub), size=min(2, len(sub)), replace=False)
        for i in pick:
            r = sub.iloc[i]
            layers = [
                (bld, dict(color="#d9d9d9", edgecolor="#bdbdbd", linewidth=0.3, kind="poly", facecolor="#d9d9d9"), "OSM building"),
                (gl[gl["inv_class"] == "in_osm"], dict(color="#9ecae1", linewidth=1.0, linestyle="--"), "GIP link (in OSM)"),
                (ways, dict(color="#525252", linewidth=1.2), "OSM walkable way"),
                (gpd.GeoDataFrame(geometry=[r.geometry], crs=links.crs), dict(color="#d7301f", linewidth=3), f"real gap ({grp})"),
            ]
            maps.example_map(fdir / f"{city}_gap_{grp}_{r.gap_id}.png", r.geometry, layers,
                             title=f"{city}: real gap, {grp}, {r.len_m:.0f} m, {r.district}",
                             note="Data: GIP 26.04 (CC BY 4.0), OSM 2026-09-22 (ODbL)")
    t("example maps written")


def unjoined_report(city):
    g = walknet.load_graph(P.graph_path(city, "original"))
    gaps = read_unjoined(city)
    tdir = C.tables()
    gl = gip.read_links(city)
    gl = gl[gl["walkable"]].explode(index_parts=False).reset_index(drop=True)
    seg = gaps.geometry.to_numpy()
    long_enough = shapely.length(seg) >= 2.0
    conf = np.full(len(gaps), np.nan)
    conf[long_enough] = INV.coverage(seg[long_enough], gl.geometry.to_numpy(), 8, 35, 1.0)
    gaps["gip_cov"] = conf
    gaps["gip_confirms"] = np.where(long_enough, conf >= 0.5, None)
    by = (gaps.groupby(["way_kind"]).agg(gaps=("gap_id", "size"), median_gap_m=("gap_m", "median"),
                                          median_saving_m=("saving_m", lambda s: float(np.nanmedian(np.where(np.isfinite(s), s, np.nan)))),
                                          no_route_within_375m=("netdist", lambda s: int((~np.isfinite(s)).sum())),
                                          gip_confirms_share=("gip_confirms", lambda s: round(float(pd.Series(s).dropna().astype(float).mean()), 3)))
          .sort_values("gaps", ascending=False).reset_index())
    by.to_csv(tdir / f"{city}_unjoined_by_kind.csv", index=False)
    pd.crosstab(gaps["district"], gaps["kind"], margins=True).to_csv(tdir / f"{city}_unjoined_by_district.csv")
    before = g.csr()
    after = UJ.csr_with_links(g, gaps)
    comp_b, comp_a = UJ.n_components(before), UJ.n_components(after)
    rng = np.random.default_rng(SEED)
    nodes = np.arange(g.n_nodes)
    city_orig = rng.choice(nodes, size=2000, replace=False)
    trips_city = UJ.trip_impact(g, before, after, city_orig, SEED)
    from scipy.spatial import cKDTree
    xy = shapely.get_coordinates(g.nodes.geometry.to_numpy())
    near = cKDTree(xy).query_ball_point(gaps[["x1", "y1"]].to_numpy(), 150.0)
    near_nodes = np.unique(np.concatenate([np.asarray(n, dtype=int) for n in near]))
    near_orig = rng.choice(near_nodes, size=min(2000, len(near_nodes)), replace=False)
    trips_near = UJ.trip_impact(g, before, after, near_orig, SEED + 1, dmin=100, dmax=800)

    def summ(df):
        fin = df[np.isfinite(df["before_m"])]
        ch = fin[fin["changed"]]
        return {"trips": int(len(df)), "share_changed": round(float(df["changed"].mean()), 4),
                "newly_reachable": int(df["newly_reachable"].sum()),
                "mean_saved_m_changed": round(float(ch["saved_m"].mean()), 1) if len(ch) else 0.0,
                "median_saved_m_changed": round(float(ch["saved_m"].median()), 1) if len(ch) else 0.0}
    impact = {"components_before": comp_b, "components_after": comp_a,
              "random_city_trips": summ(trips_city), "trips_starting_near_gaps": summ(trips_near),
              "per_gap_saving_m_median": float(np.nanmedian(np.where(np.isfinite(gaps["saving_m"]), gaps["saving_m"], np.nan))),
              "per_gap_no_route_within_375m": int((~np.isfinite(gaps["netdist"])).sum()),
              "gaps_total": int(len(gaps)),
              "gip_confirms_share": round(float(pd.Series(gaps["gip_confirms"]).dropna().astype(float).mean()), 3)}
    C.write_json(tdir / f"{city}_unjoined_impact.json", impact)
    C.log("unjoined", "components_after_join", comp_a, area=city, note=f"before {comp_b}")
    C.log("unjoined", "share_trips_changed_near", impact["trips_starting_near_gaps"]["share_changed"], area=city)
    t(f"impact: {impact}")
    pick = gaps.sample(n=min(30, len(gaps)), random_state=SEED).copy()
    p4326 = gpd.GeoSeries(pick.geometry.centroid, crs=gaps.crs).to_crs(4326)
    pick["lat"], pick["lon"] = p4326.y.round(6), p4326.x.round(6)
    pick["osm_link"] = [f"https://www.openstreetmap.org/?mlat={a}&mlon={b}#map=19/{a}/{b}" for a, b in zip(pick["lat"], pick["lon"])]
    pick["reviewer_verdict"] = ""
    pick.drop(columns="geometry").to_csv(tdir / f"{city}_unjoined_sample30.csv", index=False, encoding="utf-8-sig")
    bld = osm.read_polygons(city, "buildings")
    lines = osm.read_lines(city)
    bar = lines[lines["barrier"].fillna("").isin(UJ.BARRIERS)] if "barrier" in lines else None
    fdir = C.figures() / "unjoined"
    for r in pick.itertuples():
        layers = [
            (bld, dict(color="#d9d9d9", edgecolor="#bdbdbd", linewidth=0.3, kind="poly", facecolor="#d9d9d9"), "OSM building"),
            (gl, dict(color="#9ecae1", linewidth=1.0, linestyle="--"), "GIP walkable link"),
            (g.edges, dict(color="#525252", linewidth=1.2), "OSM walkable way"),
            (bar, dict(color="#8c510a", linewidth=1.0), "OSM barrier"),
            (gpd.GeoDataFrame(geometry=[r.geometry], crs=gaps.crs), dict(color="#d7301f", linewidth=3), "unjoined gap"),
        ]
        sv = f"{r.saving_m:.0f} m" if np.isfinite(r.saving_m) else "no route within 375 m"
        maps.example_map(fdir / f"{city}_{r.gap_id}.png", r.geometry, layers, radius=60,
                         title=f"{r.gap_id}: {r.way_kind} -> {r.target_kind}, gap {r.gap_m:.1f} m, saves {sv}",
                         note="OSM 2026-09-22 (ODbL); GIP 26.04 (CC BY 4.0) shown for reference only")
    gaps.drop(columns="geometry").to_csv(tdir / f"{city}_unjoined_all.csv", index=False)
    t("unjoined report done")


def unjoined_confirmed_impact(city):
    g = walknet.load_graph(P.graph_path(city, "original"))
    allg = pd.read_csv(C.tables() / f"{city}_unjoined_all.csv")
    conf = allg[allg["gip_confirms"].astype(str).str.lower() == "true"].reset_index(drop=True)
    before = g.csr()
    after = UJ.csr_with_links(g, conf)
    rng = np.random.default_rng(SEED)
    orig = rng.choice(np.arange(g.n_nodes), size=2000, replace=False)
    trips = UJ.trip_impact(g, before, after, orig, SEED)
    fin = trips[np.isfinite(trips["before_m"])]
    ch = fin[fin["changed"]]
    res = {"confirmed_gaps": int(len(conf)), "components_before": UJ.n_components(before),
           "components_after": UJ.n_components(after),
           "random_city_trips": {"trips": int(len(trips)), "share_changed": round(float(trips["changed"].mean()), 4),
                                 "newly_reachable": int(trips["newly_reachable"].sum()),
                                 "median_saved_m_changed": round(float(ch["saved_m"].median()), 1) if len(ch) else 0.0,
                                 "mean_saved_m_changed": round(float(ch["saved_m"].mean()), 1) if len(ch) else 0.0}}
    C.write_json(C.tables() / f"{city}_unjoined_impact_confirmed.json", res)
    C.log("unjoined", "share_trips_changed_confirmed_only", res["random_city_trips"]["share_changed"], area=city)
    t(f"confirmed-only impact: {res}")


if __name__ == "__main__":
    setup_environment()
    step, city = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "vienna")
    globals()[step](city)
