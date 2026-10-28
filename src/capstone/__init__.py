import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def _set_dll_path() -> None:
    if os.name != "nt":
        return
    prefix = Path(sys.prefix)
    dirs = [prefix / "Library" / "bin", prefix / "Library" / "mingw-w64" / "bin",
            prefix / "Library" / "usr" / "bin", prefix / "Scripts", prefix]
    have = os.environ.get("PATH", "").split(os.pathsep)
    new = [str(d) for d in dirs if d.is_dir() and str(d) not in have]
    if new:
        os.environ["PATH"] = os.pathsep.join(new + have)
    for d in dirs:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except (OSError, AttributeError):
                pass


def _set_gdal_env() -> None:
    prefix = Path(sys.prefix)
    for var, subdirs in (("GDAL_DATA", ("Library/share/gdal", "share/gdal")),
                         ("PROJ_LIB", ("Library/share/proj", "share/proj")),
                         ("PROJ_DATA", ("Library/share/proj", "share/proj"))):
        if os.environ.get(var):
            continue
        for sub in subdirs:
            p = prefix / sub
            if p.is_dir():
                os.environ[var] = str(p)
                break
    os.environ["OGR_INTERLEAVED_READING"] = "NO"
    os.environ.setdefault("PYTHONUTF8", "1")


def setup_environment() -> None:
    _set_dll_path()
    _set_gdal_env()


setup_environment()
