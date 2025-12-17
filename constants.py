from pathlib import Path
from typing import Dict

FILE_SPECS: Dict[str, Dict] = {
    "dim_date": {"file": "dim_date.csv", "parse_dates": ["date"]},
    "dim_ticker": {"file": "dim_ticker.csv", "parse_dates": None},
    "fact_prices": {"file": "fact_prices.csv", "parse_dates": ["date"]},
    "fact_features": {"file": "fact_features_daily.csv", "parse_dates": ["date"]},
    "fact_latest": {"file": "fact_latest_snapshot.csv", "parse_dates": ["last_date"]},
    "etl_metadata": {"file": "etl_metadata.csv", "parse_dates": ["run_timestamp_utc"]},
}
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
