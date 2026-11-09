# paths and small helpers
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from missingpaths import config, runlog


def interim(city: str) -> Path:
    p = config.interim_dir(city) / "rebuild"
    p.mkdir(parents=True, exist_ok=True)
    return p


def rdir(*sub: str) -> Path:
    return config.reports_dir("rebuild", *sub)


def tables() -> Path:
    return rdir("tables")


def figures() -> Path:
    return config.reports_dir("figures", "rebuild")


def write_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=default, ensure_ascii=False), encoding="utf-8")


def log(module: str, metric: str, value, area: str = "vienna", split: str = "", note: str = "") -> str:
    return runlog.log_run(f"R_{module}", metric, value, area=area, split=split, note=note)
