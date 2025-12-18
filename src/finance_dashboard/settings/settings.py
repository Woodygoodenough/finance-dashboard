from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List
from pydantic import BaseModel, Field, model_validator
from finance_dashboard.config import SOURCE, PAGES
from enum import Enum


class DatasetName(str, Enum):
    DIM_DATE = "dim_date"
    DIM_TICKER = "dim_ticker"
    FACT_PRICES = "fact_prices"
    FACT_FEATURES = "fact_features"
    FACT_LATEST = "fact_latest"
    ETL_METADATA = "etl_metadata"


class FileSpec(BaseModel):
    file_name: str
    parse_dates: List[str] | None = None


class PathsConfig(BaseModel):
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path

    @model_validator(mode="after")
    def resolve_data_dir(self):
        if isinstance(self.data_dir, str) or not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir
        return self


class UrlsConfig(BaseModel):
    data_source: str


class FileSpecsConfig(BaseModel):
    files: Dict[DatasetName, FileSpec]


def _load_file_specs(file_specs_raw: Dict[str, Dict[str, str]]) -> FileSpecsConfig:
    return FileSpecsConfig(**file_specs_raw)


class PageOverviewConfig(BaseModel):
    title: str
    icon: str
    description: str


class PagesConfig(BaseModel):
    overview: PageOverviewConfig


def _load_pages_config(pages_raw: Dict[str, Dict]) -> PagesConfig:
    return PagesConfig(**pages_raw)


def _load_paths(paths_raw: Dict[str, str]) -> PathsConfig:
    if "project_root" in paths_raw:
        raise ValueError("project_root cannot be specified")
    if "data_dir" not in paths_raw:
        raise ValueError("data_dir is required!")
    return PathsConfig(**paths_raw)


def _load_urls(urls_raw: Dict[str, str]) -> UrlsConfig:
    return UrlsConfig(**urls_raw)


# entrypoint for configuration
class AppConfig(BaseModel):
    paths: PathsConfig
    urls: UrlsConfig
    pages: PagesConfig
    file_specs: FileSpecsConfig


def load_config() -> AppConfig:
    return AppConfig(
        paths=_load_paths(SOURCE["paths"]),
        urls=_load_urls(SOURCE["urls"]),
        file_specs=_load_file_specs(SOURCE["file_specs"]),
        pages=_load_pages_config(PAGES),
    )


APP_CONFIG = load_config()
