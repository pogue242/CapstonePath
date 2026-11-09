# Data

## Folder layout

```
data/
  austria-260922.osm.pbf
  B_gip_network_ogd/gip_network_ogd.gpkg
  C_gip_reference_ogd/gip_reference_ogd.gpkg
  D_lookuptabellen/lut_*.csv
  raw/imagery/<city>/<layer>/<zoom>/<row>/<col>.jpeg
  interim/<city>/
```

## Where to download

| What | Source | Licence |
|---|---|---|
| GIP 26.04: network, reference and lookup tables | [data.gv.at](https://www.data.gv.at), search "GIP" | CC BY 4.0 |
| OpenStreetMap, Austria | [Geofabrik](https://download.geofabrik.de/europe/austria.html), saved as `austria-260922.osm.pbf` | ODbL |
| Aerial photos | `scripts/imagery_download.py` and `scripts/photo/download.py` (Stadt Wien, basemap.at) | CC BY 4.0 |
| City and district boundaries | OSMnx, or the OSM file | ODbL |
