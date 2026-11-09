# tile math and the tile downloader
from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from capstone import config

EARTH = 6378137.0
ORIGIN = math.pi * EARTH
USER_AGENT = "capstone/0.1 (CSCI 499 research; polite single-user download)"


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    attribution: str
    licence: str
    zoom: int


SOURCES = {
    "wien_ogd_2022": Source("wien_ogd_2022", "https://maps.wien.gv.at/wmts/lb/farbe/google3857/{z}/{y}/{x}.jpeg",
                            "Datenquelle: Stadt Wien – data.wien.gv.at", "CC BY 4.0", 19),
}


def tile_size_m(z: int) -> float:
    return 2 * ORIGIN / 2 ** z


def merc_to_tile(x: np.ndarray, y: np.ndarray, z: int):
    s = tile_size_m(z)
    return (np.asarray(x) + ORIGIN) / s, (ORIGIN - np.asarray(y)) / s


def tiles_for_bbox(xmin, ymin, xmax, ymax, z: int) -> list[tuple[int, int]]:
    c0, r1 = merc_to_tile(xmin, ymin, z)
    c1, r0 = merc_to_tile(xmax, ymax, z)
    return [(int(r), int(c)) for r in range(int(math.floor(r0)), int(math.floor(r1)) + 1)
            for c in range(int(math.floor(c0)), int(math.floor(c1)) + 1)]


def cache_path(src: Source, city: str, r: int, c: int) -> Path:
    return config.ROOT / "data" / "raw" / "imagery" / city / src.key / str(src.zoom) / str(r) / f"{c}.jpeg"


class Downloader:

    def __init__(self, src: Source, city: str, rate_per_s: float = 8.0, workers: int = 8):
        self.src, self.city = src, city
        self.min_gap = 1.0 / rate_per_s
        self.lock = threading.Lock()
        self.last = 0.0
        self.workers = workers
        self.stats = {"fetched": 0, "cached": 0, "failed": 0, "bytes": 0}

    def _wait(self):
        with self.lock:
            now = time.time()
            wait = self.last + self.min_gap - now
            self.last = max(now, self.last + self.min_gap)
        if wait > 0:
            time.sleep(wait)

    def fetch(self, rc):
        r, c = rc
        p = cache_path(self.src, self.city, r, c)
        if p.exists() and p.stat().st_size > 0:
            self.stats["cached"] += 1
            return
        if p.with_suffix(".missing").exists():
            self.stats["missing"] = self.stats.get("missing", 0) + 1
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        url = self.src.url.format(z=self.src.zoom, y=r, x=c)
        for attempt in range(4):
            self._wait()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                tmp = p.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(p)
                self.stats["fetched"] += 1
                self.stats["bytes"] += len(data)
                return
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    p.with_suffix(".missing").write_text("404")
                    self.stats["missing"] = self.stats.get("missing", 0) + 1
                    return
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        self.stats["failed"] += 1

    def run(self, tiles, progress_every: int = 2000):
        tiles = list(dict.fromkeys(tiles))
        t0 = time.time()
        with ThreadPoolExecutor(self.workers) as ex:
            for i, _ in enumerate(ex.map(self.fetch, tiles)):
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  tiles {i + 1:,}/{len(tiles):,} ({time.time() - t0:.0f} s) {self.stats}", flush=True)
        return dict(self.stats, tiles=len(tiles), seconds=round(time.time() - t0))


def append_manifest(entry: dict) -> None:
    p = config.reports_dir() / "data_manifest_imagery.json"
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"entries": []}
    entry = {"recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    doc["entries"].append(entry)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
