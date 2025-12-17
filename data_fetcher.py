from __future__ import annotations

import os
import pandas as pd

from constants import DATA_DIR, FILE_SPECS
from pathlib import Path


def fetch_and_store(source_root: Path) -> bool:
    """
    Download all CSVs from source_root into ./data (overwrites previous data).
    Returns success.
    """
    DATA_DIR.mkdir(exist_ok=True)

    for spec in FILE_SPECS.values():
        path = source_root / spec["file"]
        try:
            df = pd.read_csv(path, parse_dates=spec["parse_dates"])
            df.to_csv(DATA_DIR / spec["file"], index=False)
        except Exception as exc:  # noqa: BLE001
            return False
    return True


def cli_fetch() -> int:
    """
    CLI helper: fetch data from DATA_BASE_URL (env) or default GitHub Pages root.
    """
    success = fetch_and_store(DATA_DIR)
    if not success:
        print("Failed to fetch data")
        return 1
    print("Fetched data")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_fetch())
