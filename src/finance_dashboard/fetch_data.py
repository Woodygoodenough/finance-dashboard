from __future__ import annotations

from finance_dashboard.services.data_fetcher import cli_fetch


def main() -> int:
    """Entry point for the finance-fetch command."""
    return cli_fetch()


if __name__ == "__main__":
    raise SystemExit(main())
