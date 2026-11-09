# clip pedestrian areas and entrances from the OSM file
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from missingpaths import setup_environment
from missingpaths.io import boundary
from missingpaths.rebuild import osm_extra


def main(city: str) -> None:
    b = boundary.city_boundary(city)
    for name, fn in (("ped_areas", osm_extra.clip_ped_areas), ("points", osm_extra.clip_entrances)):
        t0 = time.time()
        p = fn(city, b)
        print(f"{name}: {p} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    setup_environment()
    main(sys.argv[1] if len(sys.argv) > 1 else "vienna")
