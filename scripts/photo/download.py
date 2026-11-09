# download the tiles for the planned squares
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from capstone.imagery.tiles import SOURCES, Downloader, Source, append_manifest
from capstone.rebuild import common as C


def layer_key(layer: str) -> str:
    return {"lb": "wien_ogd_2022", "basemap": "basemap_ortho"}.get(layer, f"wien_ogd_{layer}")


def main(city, plan_file, rate=8.0, layer=None):
    pdir = C.interim(city) / "photo"
    doc = json.loads((pdir / plan_file).read_text(encoding="utf-8"))
    layer = layer or doc.get("layer", "lb")
    base = SOURCES["wien_ogd_2022"]
    if layer == "basemap":
        src = Source("basemap_ortho", "https://maps.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg",
                     "Datenquelle: basemap.at", "CC BY 4.0", int(doc["zoom"]))
    else:
        url = base.url.replace("/wmts/lb/", f"/wmts/{layer}/")
        src = Source(layer_key(layer), url, base.attribution, base.licence, int(doc["zoom"]))
    tiles = [tuple(t) for v in doc["tiles"].values() for t in v]
    res = Downloader(src, city, rate_per_s=rate, workers=8).run(tiles, progress_every=1000)
    append_manifest({"source": src.key, "wmts_layer": layer, "zoom": src.zoom, "city": city, "plan": plan_file, **res,
                     "licence": src.licence, "attribution": src.attribution})
    C.log("photo_download", "tiles_fetched", res.get("fetched", 0), area=city, note=json.dumps(res))
    print("DONE", res, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
         sys.argv[4] if len(sys.argv) > 4 else None)
