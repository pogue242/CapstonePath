# pick the test districts and training groups
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from missingpaths import setup_environment
import geopandas as gpd
import pandas as pd
import yaml

from missingpaths import config
from missingpaths.rebuild import common as C
from missingpaths.rebuild import splits as S

REACH_TARGETS = {
    "footpath": 0.50, "estate_passage": 0.60, "historic_passage": 0.50, "stairs": 0.40, "other": 0.50, "all": 0.50,
}
HELD_OUT_CITIES = ["linz", "klagenfurt"]


def main(city="vienna"):
    path = config.CONFIG_DIR / "rebuild_locked.yaml"
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    gaps = gpd.read_file(C.interim(city) / "real_gaps.gpkg", layer="real_gaps")
    d = gpd.read_file(config.interim_dir(city) / "districts.gpkg")
    counts = pd.crosstab(gaps["district"], gaps["group"]).reindex(d["district"]).fillna(0).astype(int)
    area = pd.Series(d.geometry.area.to_numpy() / 1e6, index=d["district"])
    chosen, top = S.choose_test_districts(counts, area)
    top.to_csv(C.tables() / f"{city}_test_district_candidates.csv", index=False)
    train = sorted(set(d["district"]) - set(chosen))
    folds = S.greedy_folds(counts.loc[train].sum(axis=1), 5)
    doc = {
        "note": "Test areas chosen with the fixed rule in src/missingpaths/rebuild/splits.py before any training. Each test area is scored once per final model.",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": 42,
        "split_buffer_m": 200,
        "vienna": {
            "test_districts": chosen,
            "selection_rule": "3-4 districts minimising the worst |share - 0.20| over real-gap groups; "
                              ">= 3 gaps per group with >= 20 city-wide; ties: smaller area, then name",
            "test_counts_by_group": {g: int(counts.loc[chosen, g].sum()) for g in counts.columns},
            "total_counts_by_group": {g: int(counts[g].sum()) for g in counts.columns},
            "old_first_study_test_districts": ["Innere Stadt", "Neubau", "Favoriten", "Donaustadt"],
            "old_test_overlap": sorted(set(chosen) & {"Innere Stadt", "Neubau", "Favoriten", "Donaustadt"}),
            "folds": {f"fold_{k}": sorted(n for n, f in folds.items() if f == k) for k in range(5)},
            "fold_rule": "greedy balance by real-gap count (sort desc, smallest running total)",
            "reach_targets": REACH_TARGETS,
        },
        "held_out_cities": HELD_OUT_CITIES,
        "held_out_rule": "never used for training or tuning; scored once per final model",
        "previously_used_cities": ["graz", "salzburg", "innsbruck"],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    C.log("splits", "test_districts", len(chosen), area=city, note=";".join(chosen))
    print(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    setup_environment()
    main(sys.argv[1] if len(sys.argv) > 1 else "vienna")
