from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

FILE_SPECS: Dict[str, Dict] = {
    "dim_date": {"file": "dim_date.csv", "parse_dates": ["date"]},
    "dim_ticker": {"file": "dim_ticker.csv", "parse_dates": None},
    "fact_prices": {"file": "fact_prices.csv", "parse_dates": ["date"]},
    "fact_features": {"file": "fact_features_daily.csv", "parse_dates": ["date"]},
    "fact_latest": {"file": "fact_latest_snapshot.csv", "parse_dates": ["last_date"]},
    "etl_metadata": {"file": "etl_metadata.csv", "parse_dates": ["run_timestamp_utc"]},
}

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "local_cache"


def has_complete_snapshot(path: Path) -> bool:
    return (
        path.exists()
        and path.is_dir()
        and all((path / spec["file"]).exists() for spec in FILE_SPECS.values())
    )


def latest_snapshot_dir(base_dir: Path | None = None) -> Optional[Path]:
    # Backwards compatibility: prefer cache dir; fallback to any directory directly under data with full set.
    if has_complete_snapshot(CACHE_DIR):
        return CACHE_DIR
    base_dir = base_dir or DATA_DIR
    if not base_dir.exists():
        return None
    snaps = sorted([p for p in base_dir.iterdir() if p.is_dir()], reverse=True)
    for snap in snaps:
        if has_complete_snapshot(snap):
            return snap
    return None


def discover_local_base_url() -> Optional[str]:
    """Return file:// URI for a valid local snapshot under ./data/."""
    snap = latest_snapshot_dir()
    if not snap:
        return None
    return snap.resolve().as_uri()


def fetch_and_store(source_root: str) -> Tuple[Optional[Path], List[str]]:
    """
    Download all CSVs from source_root into ./data/local_cache (overwrites previous cache).
    Returns (path_to_snapshot, errors).
    """
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR

    errors: List[str] = []
    for spec in FILE_SPECS.values():
        url = f"{source_root.rstrip('/')}/{spec['file']}"
        try:
            df = pd.read_csv(url, parse_dates=spec["parse_dates"])
            df.to_csv(dest / spec["file"], index=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec['file']}: {exc}")

    if errors:
        try:
            for f in dest.iterdir():
                f.unlink()
        except Exception:
            pass
        return None, errors
    return dest, errors


def ensure_local_data(
    source_root: str, manual_refresh: bool = False
) -> Tuple[Optional[str], List[str]]:
    """
    Ensure a recent local snapshot exists. Auto-refresh if older than 24h or on manual trigger.
    Returns (file:// URL to latest snapshot, errors).
    """
    errors: List[str] = []
    latest = latest_snapshot_dir()
    needs_auto_refresh = False
    if latest:
        age = datetime.utcnow() - datetime.utcfromtimestamp(latest.stat().st_mtime)
        needs_auto_refresh = age > timedelta(hours=24)

    if manual_refresh or needs_auto_refresh or latest is None:
        new_snap, errs = fetch_and_store(source_root)
        errors.extend(errs)
        if new_snap:
            latest = new_snap

    if latest:
        return latest.resolve().as_uri(), errors
    return None, errors


def cli_fetch() -> int:
    """
    CLI helper: fetch data from DATA_BASE_URL (env) or default GitHub Pages root.
    """
    source = os.getenv(
        "DATA_BASE_URL", "https://Woodygoodenough.github.io/finance-data-ETL/data"
    )
    snap, errs = fetch_and_store(source)
    if errs:
        print("Errors:\n- " + "\n- ".join(errs))
        return 1
    print(f"Fetched snapshot to {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_fetch())
