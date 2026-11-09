# download Vienna aerial photo tiles
import json
import sys
from pathlib import Path

from capstone import setup_environment
from capstone import runlog
from capstone.imagery.tiles import SOURCES, append_manifest, cache_path


def main():
    city, plan, purpose = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    src_key = sys.argv[4] if len(sys.argv) > 4 else "wien_ogd_2022"
    src = SOURCES[src_key]
    from capstone.imagery.tiles import Downloader
    tiles = [tuple(t) for t in json.loads(plan.read_text())]
    d = Downloader(src, city)
    stats = d.run(tiles)
    size = sum(cache_path(src, city, r, c).stat().st_size for r, c in tiles if cache_path(src, city, r, c).exists())
    append_manifest({"city": city, "purpose": purpose, "source": src.key, "url_template": src.url,
                     "licence": src.licence, "attribution": src.attribution, "zoom": src.zoom,
                     "tiles_planned": len(tiles), **stats, "bytes_on_disk_for_plan": size,
                     "folder": f"data/raw/imagery/{city}/{src.key}/{src.zoom}/"})
    runlog.log_run("imagery_download", "tiles_failed", stats["failed"], area=city,
                   note=f"{purpose}; {stats}")
    print(stats)


if __name__ == "__main__":
    setup_environment()
    main()
