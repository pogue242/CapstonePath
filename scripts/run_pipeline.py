import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from missingpaths import setup_environment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("city", nargs="?", default="vienna")
    a = ap.parse_args()
    from missingpaths import pipeline as P
    s, c = a.step, a.city
    if s == "m0":
        from missingpaths.tools import prepare_data
        res = prepare_data.main()
    elif s == "m1":
        res = P.m1_clip(c)
    elif s == "m2":
        res = P.m2_labels(c)
    elif s == "m3":
        res = P.m3_match(c)
    else:
        ap.error(f"unknown step {s}")
    print(json.dumps(res, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    setup_environment()
    main()
