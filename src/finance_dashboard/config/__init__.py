from importlib.resources import files
import yaml


def load_yaml(package: str, filename: str) -> dict:
    p = files(package).joinpath(filename)
    return yaml.safe_load(p.read_text("utf-8")) or {}


SOURCE = load_yaml("finance_dashboard.config", "source.yml")
PAGES = load_yaml("finance_dashboard.config", "pages.yml")
