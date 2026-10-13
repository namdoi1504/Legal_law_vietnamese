"""Build a lexical preview only; never overwrite the active paired indexes."""

from scripts.build_index import build
from configs.settings import ROOT


if __name__ == "__main__":
    build(ROOT / "data/raw/labor_law_2019.html", activate=False, lexical_only=True)
