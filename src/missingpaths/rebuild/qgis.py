# write a QGIS project
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

ORTHO_URL = "https://maps.wien.gv.at/wmts/lb/farbe/google3857/%7Bz%7D/%7By%7D/%7Bx%7D.jpeg"


def _symbol(geom: str, rgba: str, width: float = 0.6) -> str:
    if geom == "Line":
        cls, opts = "SimpleLine", f'<Option name="line_color" value="{rgba}" type="QString"/><Option name="line_width" value="{width}" type="QString"/>'
        st = "line"
    elif geom == "Polygon":
        cls, opts = "SimpleFill", f'<Option name="color" value="{rgba}" type="QString"/><Option name="outline_color" value="{rgba}" type="QString"/>'
        st = "fill"
    else:
        cls, opts = "SimpleMarker", f'<Option name="color" value="{rgba}" type="QString"/><Option name="size" value="2" type="QString"/>'
        st = "marker"
    return (f'<symbol type="{st}" name="0" alpha="1" clip_to_extent="1" force_rhr="0">'
            f'<layer class="{cls}" enabled="1" locked="0" pass="0"><Option type="Map">{opts}</Option></layer></symbol>')


def _renderer(geom: str, rgba: str, width: float, cat_attr: str | None, cats: dict | None) -> str:
    if not cat_attr:
        return f'<renderer-v2 type="singleSymbol" symbollevels="0" forceraster="0" enableorderby="0"><symbols>{_symbol(geom, rgba, width)}</symbols></renderer-v2>'
    cs, ss = [], []
    for i, (val, col) in enumerate(cats.items()):
        cs.append(f'<category render="true" symbol="{i}" value="{escape(str(val))}" label="{escape(str(val))}"/>')
        ss.append(_symbol(geom, col, width).replace('name="0"', f'name="{i}"', 1))
    return (f'<renderer-v2 type="categorizedSymbol" attr="{cat_attr}" symbollevels="0" forceraster="0" enableorderby="0">'
            f'<categories>{"".join(cs)}</categories><symbols>{"".join(ss)}</symbols></renderer-v2>')


def write_project(path: Path, gpkg_name: str, layers: list[dict], crs: str = "EPSG:31256") -> None:
    tree, maps = [], []
    for i, L in enumerate(layers):
        lid = f"L{i}_{L['name']}"
        src = f"./{gpkg_name}|layername={L['name']}"
        chk = "Qt::Checked" if L.get("visible", True) else "Qt::Unchecked"
        tree.append(f'<layer-tree-layer id="{lid}" name="{L["name"]}" source="{escape(src)}" providerKey="ogr" checked="{chk}" expanded="0"/>')
        maps.append(
            f'<maplayer type="vector" geometry="{L["geom"]}"><id>{lid}</id><datasource>{escape(src)}</datasource>'
            f'<layername>{L["name"]}</layername><provider encoding="UTF-8">ogr</provider>'
            f'<srs><spatialrefsys><authid>{crs}</authid></spatialrefsys></srs>'
            f'{_renderer(L["geom"], L["color"], L.get("width", 0.6), L.get("cat_attr"), L.get("cats"))}</maplayer>')
    ortho_src = f"type=xyz&amp;url={ORTHO_URL}&amp;zmax=20&amp;zmin=0"
    tree.append(f'<layer-tree-layer id="ortho" name="Orthofoto 2022 (Stadt Wien)" source="{ortho_src}" providerKey="wms" checked="Qt::Checked" expanded="0"/>')
    maps.append(f'<maplayer type="raster"><id>ortho</id><datasource>{ortho_src}</datasource>'
                f'<layername>Orthofoto 2022 (Stadt Wien)</layername><provider>wms</provider>'
                f'<srs><spatialrefsys><authid>EPSG:3857</authid></spatialrefsys></srs></maplayer>')
    xml = ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
           f'<qgis projectname="missingpaths" version="3.34.0">'
           f'<projectCrs><spatialrefsys><authid>{crs}</authid></spatialrefsys></projectCrs>'
           f'<layer-tree-group>{"".join(tree)}</layer-tree-group>'
           f'<projectlayers>{"".join(maps)}</projectlayers></qgis>\n')
    path.write_text(xml, encoding="utf-8")
