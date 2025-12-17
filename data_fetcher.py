from __future__ import annotations
import pandas as pd

from constants import DATA_DIR, FILE_SPECS, SOURCE_URL


def fetch_and_store(source_url: str) -> bool:
    """
    Download all CSVs from source_url into ./data (overwrites previous data).
    Returns success.
    """
    DATA_DIR.mkdir(exist_ok=True)

    for spec in FILE_SPECS.values():
        url = f"{source_url}/{spec['file']}"
        try:
            df = pd.read_csv(url, parse_dates=spec["parse_dates"])
            df.to_csv(DATA_DIR / spec["file"], index=False)
        except Exception as exc:  # noqa: BLE001
            return False
    return True


def cli_fetch() -> int:
    """
    CLI helper: fetch data from SOURCE_URL.
    """

    success = fetch_and_store(SOURCE_URL)
    if not success:
        print("Failed to fetch data")
        return 1
    print("Fetched data")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_fetch())
