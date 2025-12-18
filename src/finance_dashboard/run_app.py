from __future__ import annotations

import sys
from pathlib import Path


def _get_app_path() -> Path:
    """Get the path to app.py relative to this file."""
    return Path(__file__).parent / "app.py"


def main() -> None:
    """Entry point for running the Streamlit app (development mode)."""
    import streamlit.web.cli as stcli

    app_path = _get_app_path()
    # Set up sys.argv as if streamlit was called directly
    sys.argv = ["streamlit", "run", str(app_path.resolve())] + sys.argv[1:]
    stcli.main()


def main_prod() -> None:
    """Entry point for running the Streamlit app (production mode with server options)."""
    import streamlit.web.cli as stcli

    app_path = _get_app_path()
    # Set up sys.argv with server address and port options
    sys.argv = [
        "streamlit",
        "run",
        str(app_path.resolve()),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
    ] + sys.argv[1:]
    stcli.main()


if __name__ == "__main__":
    main()
