from __future__ import annotations

import os
import pandas as pd

from constants import DATA_DIR, FILE_SPECS


def fetch_and_store(source_root: str) -> bool:
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
    source = os.getenv(
        "DATA_BASE_URL", "https://Woodygoodenough.github.io/finance-data-ETL/data"
    )
    success = fetch_and_store(source)
    if not success:
        print("Failed to fetch data")
        return 1
    print("Fetched data")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_fetch())
