# %%
from __future__ import annotations
import pandas as pd

from finance_dashboard.settings import APP_CONFIG, DatasetName


def fetch_and_store(source_url: str) -> bool:
    APP_CONFIG.paths.data_dir.mkdir(exist_ok=True)

    for name in DatasetName:
        spec = APP_CONFIG.file_specs.files[name]
        url = f"{source_url}/{spec.file_name}"
        print(f"Fetching {spec.file_name} from {url}")
        df = pd.read_csv(url, parse_dates=spec.parse_dates)
        output_path = APP_CONFIG.paths.data_dir / spec.file_name
        print(f"Writing {len(df)} rows to {output_path}")
        df.to_csv(output_path, index=False)
        print(f"Successfully wrote {spec.file_name}")

    return True


def cli_fetch() -> int:
    success = fetch_and_store(APP_CONFIG.urls.data_source)
    if not success:
        print("Failed to fetch data")
        return 1
    print("Fetched data")
    return 0
