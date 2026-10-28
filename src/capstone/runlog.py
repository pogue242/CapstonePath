from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from capstone.config import CONFIG_DIR, file_sha256, reports_dir

COLUMNS = ["timestamp", "run_id", "module", "area", "split", "params_hash", "locked_hash",
           "data_manifest_hash", "metric_name", "metric_value", "note"]


def _hash_or_blank(path: Path) -> str:
    return file_sha256(path)[:12] if path.exists() else ""


def new_run_id(module: str) -> str:
    return f"{module}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


def log_run(module: str, metric_name: str, metric_value, *, area: str = "", split: str = "",
            note: str = "", run_id: str | None = None, path: Path | None = None) -> str:
    path = path or reports_dir() / "runs.csv"
    run_id = run_id or new_run_id(module)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id, "module": module, "area": area, "split": split,
        "params_hash": _hash_or_blank(CONFIG_DIR / "params.yaml"),
        "locked_hash": _hash_or_blank(CONFIG_DIR / "locked.yaml"),
        "data_manifest_hash": _hash_or_blank(reports_dir() / "data_manifest.json"),
        "metric_name": metric_name,
        "metric_value": "" if metric_value is None else metric_value,
        "note": note,
    }
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return run_id
