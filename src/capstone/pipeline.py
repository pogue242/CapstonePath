from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from capstone import config, runlog
from capstone.io import boundary, gip, osm
from capstone.labels import build as labels_build
from capstone.labels import match as match_mod
from capstone.labels import stations

EXPECTED_VIENNA = {
    "links_total": 161395, "walkable_links": 151670, "ped_only_links": 107097,
    "passage_features": 1974, "stairs_features": 1426, "indoor_vertical_features": 734,
    "osm_ped_ways": 91415, "osm_building_passage": 2606, "osm_steps": 6841,
}


def tables_dir():
    return config.reports_dir("tables")


def m1_clip(city: str, osm_layers=("lines", "polygons")) -> dict:
    b = boundary.city_boundary(city)
    gip.clip_gip(city, b)
    if "lines" in osm_layers:
        osm.clip_osm_lines(city, b)
    if "polygons" in osm_layers:
        osm.clip_osm_polygons(city, b)
    return {"boundary_km2": round(float(b.to_crs(config.metric_crs(city)).area.iloc[0]) / 1e6, 2)}


def reproduction_check(city: str = "vienna") -> dict:
    from capstone.labels import micro
    links = gip.read_links(city)
    g = gip.summary_counts(links)
    f = micro.aggregate_features(micro.micro_links(links))
    fc = f["cls"].value_counts()
    o = osm.summary_counts(osm.read_lines(city))
    actual = {
        "links_total": g["links_total"], "walkable_links": g["walkable_links"],
        "ped_only_links": g["ped_only_links"],
        "passage_features": int(fc.get("passage", 0)), "stairs_features": int(fc.get("stairs", 0)),
        "indoor_vertical_features": int(fc.get("indoor_vertical", 0)),
        "osm_ped_ways": o["ped_ways"], "osm_building_passage": o["tunnel=building_passage"],
        "osm_steps": o["steps"],
    }
    ok = all(actual[k] == v for k, v in EXPECTED_VIENNA.items())
    res = {"actual": actual, "expected": EXPECTED_VIENNA, "all_match": ok,
           "extra": {**g, **o}}
    (tables_dir() / "reproduction_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    runlog.log_run("M0_reproduction", "all_match", int(ok), area=city,
                   note=";".join(f"{k}={v}" for k, v in actual.items()))
    return res


def district_regional_code_check(city: str, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    links = gip.read_links(city)
    links = links[links["walkable"]]
    links["district"] = labels_build.assign_district(links, districts)
    t = (links.dropna(subset=["district"]).groupby(["regional_code", "district"]).size()
         .rename("links").reset_index())
    tot = t.groupby("regional_code")["links"].transform("sum")
    t["share"] = (t["links"] / tot).round(4)
    top = t.sort_values("links", ascending=False).drop_duplicates("regional_code")
    return top.sort_values("regional_code").reset_index(drop=True)


def test_district_check(feats: pd.DataFrame) -> dict:
    lk = config.locked()
    tgt = labels_build.target_features(feats)
    test = list(lk["vienna_test_districts"])
    backups = list(lk["test_district_backup_order"])
    need = int(lk["min_test_features_per_stratum"])
    history = []
    while True:
        sub = tgt[tgt["district"].isin(test)]
        counts = {s: int((sub["stratum"] == s).sum()) for s in ("passage_historic", "passage_estate", "stairs")}
        passed = all(v >= need for v in counts.values())
        history.append({"districts": list(test), "counts": counts, "passed": passed})
        if passed or not backups:
            break
        test.append(backups.pop(0))
    return {"final_test_districts": test, "passed": passed, "history": history,
            "changed_from_locked": test != list(lk["vienna_test_districts"])}


def m2_labels(city: str) -> dict:
    has_districts = bool(config.city(city).get("districts"))
    d = boundary.districts(city) if has_districts else None
    feats = labels_build.build_micro_features(city, districts=d)
    summ = labels_build.summary_table(feats)
    summ.to_csv(tables_dir() / f"{city}_micro_summary.csv", index=False)
    agree = stations.agreement_report(feats[feats["cls"].isin(["passage", "stairs"])])
    agree.to_csv(tables_dir() / f"{city}_station_filter_agreement.csv", index=False)
    tgt = labels_build.target_features(feats)
    out = {"features": feats["cls"].value_counts().to_dict(),
           "usable_targets": tgt["stratum"].value_counts().to_dict(),
           "usable_total": int(len(tgt))}
    if d is not None:
        d.to_file(config.interim_dir(city) / "districts.gpkg", layer="districts", driver="GPKG")
        per = pd.crosstab(tgt["district"], tgt["stratum"], margins=True)
        per.to_csv(tables_dir() / f"{city}_targets_by_district.csv")
        rc = district_regional_code_check(city, d)
        rc.to_csv(tables_dir() / f"{city}_regional_code_vs_district.csv", index=False)
        out["n_districts"] = int(len(d))
        out["regional_code_min_share"] = float(rc["share"].min()) if len(rc) else None
    if city == "vienna":
        chk = test_district_check(feats)
        (tables_dir() / "test_district_check.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
        out["test_district_check"] = chk
    all_row = agree[agree["class"] == "all"].iloc[0]
    out["general_rule_catches_subnet_share"] = all_row["general_catches_share_of_subnet"]
    runlog.log_run("M2_labels", "usable_target_features", out["usable_total"], area=city,
                   note=json.dumps(out["usable_targets"]))
    return out


def walkable_osm(city: str) -> gpd.GeoDataFrame:
    from capstone.graph.walknet import walkable_mask
    lines = osm.read_lines(city)
    w = lines[walkable_mask(lines)].explode(index_parts=False).reset_index(drop=True)
    return w[w.geometry.notna() & ~w.geometry.is_empty]


def m3_match(city: str, write_review: bool = True) -> dict:
    lk = config.locked()["match"]
    tol, ang, step = lk["tol_m"], lk["angle_deg"], lk["step_m"]
    mmin, pmin = lk["matched_min_coverage"], lk["partial_min_coverage"]
    feats = labels_build.read_features(city)
    feats = feats[feats["cls"].isin(["passage", "stairs"])].reset_index(drop=True)
    feats["target"] = ~feats["is_station"]
    ways = walkable_osm(city)
    ped_ways = ways[osm.pedestrian_mask(ways)].reset_index(drop=True)

    status, hits = match_mod.match_features(feats, ways, tol, ang, step, mmin, pmin)
    dom = match_mod.dominant_way(hits, len(feats))
    res = feats.drop(columns="geometry").merge(status, on="feature_id")
    for col in ("osm_id", "highway", "tunnel", "covered"):
        vals = ways[col].to_numpy(dtype=object)
        res[f"osm_{col}"] = np.where(dom >= 0, vals[np.maximum(dom, 0)], None)
    res = gpd.GeoDataFrame(res, geometry=feats.geometry.to_numpy(), crs=feats.crs)

    gl = gip.read_links(city)
    gl = gl[gl["walkable"]].explode(index_parts=False).reset_index(drop=True)
    om = ways[(ways["highway"] == "steps") | (ways["tunnel"] == "building_passage")].copy()
    om["kind"] = np.where(om["highway"] == "steps", "steps", "building_passage")
    rev = match_mod.reverse_match(om.reset_index(drop=True), gl, tol, ang, step, mmin, pmin)
    rev = gpd.GeoDataFrame(rev, geometry=om.geometry.to_numpy(), crs=om.crs)

    plan = match_mod.deletion_plan(feats, status, hits, ways, tol)

    out_gpkg = config.interim_dir(city) / "match.gpkg"
    tmp = out_gpkg.with_suffix(".tmp.gpkg")
    res.to_file(tmp, layer="gip_features", driver="GPKG")
    rev.to_file(tmp, layer="osm_micro", driver="GPKG")
    tmp.replace(out_gpkg)
    plan.to_csv(config.interim_dir(city) / "deletion_plan.csv", index=False)
    res.drop(columns="geometry").to_csv(tables_dir() / f"{city}_match_features.csv", index=False)

    tgt = res[res["target"]]
    ct = pd.crosstab(tgt["stratum"], tgt["status"], margins=True)
    ct.to_csv(tables_dir() / f"{city}_match_status_by_stratum.csv")
    st = res[~res["target"]]
    pd.crosstab(st["cls"], st["status"], margins=True).to_csv(tables_dir() / f"{city}_match_status_stations.csv")
    pd.crosstab(rev["kind"], rev["status"], margins=True).to_csv(tables_dir() / f"{city}_osm_micro_vs_gip.csv")

    sens = match_mod.sensitivity_table(feats[feats["target"]].reset_index(drop=True),
                                       {"osm_walkable": ways, "osm_pedestrian_prototype": ped_ways},
                                       [5, 8, 12], [20, 35, 50], step, mmin, pmin)
    sens.to_csv(tables_dir() / f"match_sensitivity_{city}.csv", index=False)

    out = {"status_targets": tgt["status"].value_counts().to_dict(),
           "osm_micro_vs_gip": rev.groupby(["kind", "status"]).size().rename("n").reset_index()
           .to_dict(orient="records"),
           "deletion_ways_drop": int((plan["action"] == "drop").sum()),
           "deletion_ways_cut": int((plan["action"] == "cut").sum())}
    if city == "vienna" and write_review:
        sample = match_mod.stratified_review_sample(res, config.seed(), res.crs)
        rdir = config.reports_dir("review")
        path = rdir / "review_gip_only_sample.csv"
        if path.exists():
            path = rdir / "review_gip_only_sample.regenerated.csv"
        sample.to_csv(path, index=False, encoding="utf-8-sig")
        out["review_rows"] = sample["stratum"].value_counts().to_dict()
        out["review_path"] = str(path.relative_to(config.ROOT))
    runlog.log_run("M3_match", "gip_only_targets", int((tgt["status"] == "gip_only").sum()),
                   area=city, note=json.dumps(out["status_targets"]))
    return out
