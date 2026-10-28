# Data

The data isn't in this repository because the files are too large for GitHub. Everything used is public, and the sources are listed below. I'm still working out how to share the processed data and results, and will upload them later.

## Folder layout

The code finds the input files by name anywhere under `data/`, so the subfolder names don't matter. This is the layout I use:

```
data/
  austria-260922.osm.pbf               OpenStreetMap extract for Austria
  B_gip_network_ogd/
    gip_network_ogd.gpkg               GIP network
  C_gip_reference_ogd/
    gip_reference_ogd.gpkg             GIP reference layers
  D_lookuptabellen/
    lut_form_of_way_*.csv              GIP lookup tables (keep all the lut_*.csv files together)
  interim/<city>/                      written by the pipeline
```

## Where to download

| What | Source | Licence |
|---|---|---|
| GIP (Austria's official transport graph), release 26.04: the network, reference and lookup-table packages | [data.gv.at](https://www.data.gv.at), search for "GIP" (Intermodales Verkehrsreferenzsystem Österreich). The routing export package isn't needed. | CC BY 4.0 |
| OpenStreetMap, Austria extract | [Geofabrik](https://download.geofabrik.de/europe/austria.html). The current file is `austria-latest.osm.pbf`: rename it to `austria-260922.osm.pbf`, or change `osm_pbf` in `config/cities.yaml`. A newer extract gives slightly different counts. | ODbL |
| City and district boundaries | No manual download. They are geocoded through OSMnx, or read from the OSM extract when offline. | ODbL |
