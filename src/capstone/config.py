from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("CAPSTONE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
DOCS_DIR = ROOT / "docs"
EXCLUDED_DATA_PARTS = ("A_routingexport",)


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def locked() -> dict:
    return _read_yaml(CONFIG_DIR / "locked.yaml")


@lru_cache(maxsize=None)
def params() -> dict:
    return _read_yaml(CONFIG_DIR / "params.yaml")


@lru_cache(maxsize=None)
def cities_cfg() -> dict:
    return _read_yaml(CONFIG_DIR / "cities.yaml")


def city(key: str) -> dict:
    c = dict(cities_cfg()["cities"][key])
    c["key"] = key
    return c


def seed() -> int:
    return int(locked()["seed"])


def file_sha256(path: Path, normalize_newlines: bool = True) -> str:
    data = Path(path).read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def glob_one(pattern: str, root: Path = ROOT) -> Path:
    hits = [p for p in root.glob(pattern)
            if not any(part in str(p) for part in EXCLUDED_DATA_PARTS)]
    if not hits:
        raise FileNotFoundError(f"No file matches {pattern!r} under {root}")
    if len(hits) > 1:
        raise RuntimeError(f"{len(hits)} files match {pattern!r}: {hits}. Resolve by hand.")
    return hits[0]


def data_path(key: str) -> Path:
    d = cities_cfg()["data"]
    glob_key = key if key in d else f"{key}_glob"
    return glob_one(d[glob_key])


def osm_pbf_for(city_key: str) -> Path:
    own = cities_cfg()["cities"].get(city_key, {}).get("osm_pbf")
    return glob_one(own) if own else data_path("osm_pbf")


def lut_dir() -> Path:
    return glob_one(cities_cfg()["data"]["gip_lut_dir_glob"]).parent


def interim_dir(city_key: str) -> Path:
    p = ROOT / params()["paths"]["cache_dir"] / city_key
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(*sub: str) -> Path:
    p = ROOT / params()["paths"]["reports_dir"]
    for s in sub:
        p = p / s
    p.mkdir(parents=True, exist_ok=True)
    return p


def metric_crs(city_key: str) -> str:
    return city(city_key)["crs"]
