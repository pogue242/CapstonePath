from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pyogrio

from capstone import config

PACKAGES = ["numpy", "pandas", "scipy", "geopandas", "shapely", "pyogrio", "networkx", "osmnx",
            "scikit-learn", "lightgbm", "fastapi", "pyyaml", "matplotlib"]


def unzip_pending(data_dir: Path) -> list[str]:
    log = []
    existing = {p.name for p in data_dir.rglob("*") if p.is_file()}
    for z in data_dir.rglob("*.zip"):
        if any(part in str(z) for part in config.EXCLUDED_DATA_PARTS):
            log.append(f"skip (package A): {z}")
            continue
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            if all(Path(m).name in existing for m in members):
                log.append(f"already extracted: {z.name}")
                continue
            target = z.with_suffix("")
            zf.extractall(target)
            log.append(f"extracted {z.name} -> {target}")
    return log


def build_manifest() -> dict:
    net = config.data_path("gip_network")
    ref = config.data_path("gip_reference")
    pbf = config.data_path("osm_pbf")
    lut = config.lut_dir()
    version = pyogrio.read_dataframe(net, layer="version", read_geometry=False)
    cfg = config.cities_cfg()["data"]

    def rel(p: Path) -> str:
        return p.relative_to(config.ROOT).as_posix()

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    versions["gdal"] = pyogrio.__gdal_version_string__
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gip_release": cfg["gip_release"],
        "gip_version_layer": version.astype(str).to_dict(orient="records"),
        "osm_snapshot": cfg["osm_snapshot"],
        "files": {
            "gip_network": {"path": rel(net), "bytes": net.stat().st_size},
            "gip_reference": {"path": rel(ref), "bytes": ref.stat().st_size},
            "osm_pbf": {"path": rel(pbf), "bytes": pbf.stat().st_size},
            "gip_lut_dir": {"path": rel(lut), "files": sorted(p.name for p in lut.glob("lut_*.csv"))},
        },
        "package_versions": versions,
    }


def main() -> dict:
    data_dir = config.ROOT / "data"
    for line in unzip_pending(data_dir):
        print(line)
    manifest = build_manifest()
    out = config.reports_dir() / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
