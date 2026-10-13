"""Compatibility entry point: publish both indexes from one validated source snapshot."""

from scripts.build_index import build
from configs.settings import ROOT


if __name__ == "__main__":
    build(ROOT / "data/raw/labor_law_2019.html")
