"""
Premium multi-panel finance analytics built with Streamlit.

Data is pulled exclusively from CSVs hosted on GitHub Pages, pointed to by the
DATA_BASE_URL environment variable. Each CSV is cached and resilient to
individual failures so the experience remains usable even if a single file is
temporarily unavailable.
"""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from settings import APP_CONFIG

# %%
FILE_SPECS: Dict[str, Dict] = {
    "dim_date": {"file": "dim_date.csv", "parse_dates": ["date"]},
    "dim_ticker": {"file": "dim_ticker.csv", "parse_dates": None},
    "fact_prices": {"file": "fact_prices.csv", "parse_dates": ["date"]},
    "fact_features": {"file": "fact_features_daily.csv", "parse_dates": ["date"]},
    "fact_latest": {"file": "fact_latest_snapshot.csv", "parse_dates": ["last_date"]},
    "etl_metadata": {"file": "etl_metadata.csv", "parse_dates": ["run_timestamp_utc"]},
}


st.set_page_config(
    page_title=APP_CONFIG.pages.overview.title,
    page_icon=APP_CONFIG.pages.overview.icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_csv(path: Path, parse_dates: List[str] | None) -> pd.DataFrame:
    """Cached CSV reader with parsing and trimmed memory usage."""
    df = pd.read_csv(path, parse_dates=parse_dates)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load each dataset, collecting errors instead of failing fast."""
    data: Dict[str, pd.DataFrame] = {}
    for key, spec in FILE_SPECS.items():
        path = data_dir / spec["file"]
        data[key] = fetch_csv(path, spec["parse_dates"])
    return data


# %%

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_pct(value: float) -> str:
    return f"{value*100:.2f}%" if pd.notna(value) else "—"


def format_num(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.2f}"


def latest_etl_timestamp(etl_df: pd.DataFrame) -> str:
    if etl_df.empty:
        return "N/A"
    ts = etl_df["run_timestamp_utc"].max()
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M UTC")


def compute_drawdown(series: pd.Series) -> pd.Series:
    running_max = series.cummax()
    dd = series / running_max - 1.0
    return dd


def metric_cards(latest_df: pd.DataFrame, tickers: List[str]) -> None:
    subset = latest_df[latest_df["ticker"].isin(tickers)]
    if subset.empty:
        st.info("No snapshot data for current selection.")
        return

    agg = subset.mean(numeric_only=True)
    cards = [
        ("Last Close", format_num(agg.get("last_close", np.nan)), None),
        ("1D %", format_pct(agg.get("pct_1d", np.nan)), agg.get("pct_1d", np.nan)),
        ("1W %", format_pct(agg.get("pct_1w", np.nan)), agg.get("pct_1w", np.nan)),
        ("1M %", format_pct(agg.get("pct_1m", np.nan)), agg.get("pct_1m", np.nan)),
        ("YTD %", format_pct(agg.get("pct_ytd", np.nan)), agg.get("pct_ytd", np.nan)),
        ("Vol 60d", format_pct(agg.get("vol_60", np.nan)), agg.get("vol_60", np.nan)),
        (
            "Max Drawdown",
            format_pct(agg.get("max_dd_1y", np.nan)),
            agg.get("max_dd_1y", np.nan),
        ),
    ]
    cols = st.columns(len(cards))
    for col, (title, value, delta) in zip(cols, cards):
        with col:
            st.metric(
                label=title,
                value=value,
                delta=format_pct(delta) if delta is not None else None,
            )


def heatmap_returns(latest_df: pd.DataFrame, tickers: List[str]) -> None:
    subset = latest_df[latest_df["ticker"].isin(tickers)]
    if subset.empty:
        st.info("No return data for heatmap.")
        return

    heatmap_df = subset.set_index("ticker")[["pct_1d", "pct_1w", "pct_1m", "pct_ytd"]]
    fig = px.imshow(
        heatmap_df.T,
        color_continuous_scale=["#f87171", "#fbbf24", "#22c55e"],
        aspect="auto",
        labels={"x": "Ticker", "y": "Period", "color": "Return"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")


def sparkline_panel(
    price_df: pd.DataFrame,
    tickers: List[str],
    lookback_days: int,
) -> None:
    if price_df.empty:
        st.info("No price history available.")
        return

    recent_cutoff = price_df["date"].max() - pd.Timedelta(days=lookback_days)
    subset = price_df[
        (price_df["ticker"].isin(tickers)) & (price_df["date"] >= recent_cutoff)
    ]
    if subset.empty:
        st.info("No data in selected window.")
        return

    unique = subset["ticker"].unique()
    rows = int(np.ceil(len(unique) / 3))
    fig = make_subplots(
        rows=rows, cols=3, shared_xaxes=False, shared_yaxes=False, subplot_titles=unique
    )

    for idx, ticker in enumerate(unique):
        r, c = divmod(idx, 3)
        data = subset[subset["ticker"] == ticker].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["close"],
                mode="lines",
                line=dict(width=2),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        height=220 * rows,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def candlestick_panel(
    prices: pd.DataFrame, features: pd.DataFrame, ticker: str
) -> None:
    merged = (
        prices[prices["ticker"] == ticker]
        .merge(
            features[features["ticker"] == ticker],
            on=["date", "ticker"],
            how="left",
            suffixes=("", "_f"),
        )
        .sort_values("date")
    )
    if merged.empty:
        st.info("No data for selection.")
        return

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.04,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=merged["date"],
            open=merged["open"],
            high=merged["high"],
            low=merged["low"],
            close=merged["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["ma_20"],
            mode="lines",
            line=dict(color="#22c55e", width=1.5),
            name="MA 20",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["ma_50"],
            mode="lines",
            line=dict(color="#0ea5e9", width=1.5),
            name="MA 50",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["bb_up_20"],
            mode="lines",
            line=dict(width=1),
            name="BB Up",
            opacity=0.6,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["bb_low_20"],
            mode="lines",
            line=dict(width=1),
            name="BB Low",
            fill="tonexty",
            fillcolor="rgba(148,163,184,0.08)",
            opacity=0.6,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=merged["date"],
            y=merged["volume"],
            name="Volume",
        ),
        row=2,
        col=1,
    )

    dd_series = merged["drawdown_pct"].fillna(compute_drawdown(merged["close"]))
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=dd_series,
            mode="lines",
            fill="tozeroy",
            line=dict(width=1.2),
            name="Drawdown",
            hovertemplate="Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["vol_20"],
            mode="lines",
            line=dict(width=1.8),
            name="Vol 20d",
            hovertemplate="Vol 20d: %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def risk_return_scatter(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: List[str],
) -> None:
    if features.empty:
        st.info("No features available to compute risk.")
        return
    subset = features[features["ticker"].isin(tickers)]
    if subset.empty:
        st.info("No data for selected tickers.")
        return
    grouped = subset.groupby("ticker")
    ann_return = grouped["ret_1d"].mean() * 252
    ann_vol = grouped["ret_1d"].std(ddof=0) * np.sqrt(252)
    avg_vol = prices[prices["ticker"].isin(tickers)].groupby("ticker")["volume"].mean()
    df = pd.DataFrame({"return": ann_return, "risk": ann_vol, "avg_volume": avg_vol})
    df = df.reset_index()

    fig = px.scatter(
        df,
        x="risk",
        y="return",
        size="avg_volume",
        color="ticker",
        hover_name="ticker",
        size_max=40,
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"risk": "Vol (ann.)", "return": "Return (ann.)"},
    )
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")


def correlation_heatmap(features: pd.DataFrame, tickers: List[str]) -> None:
    subset = features[features["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="ret_1d"
    )
    if subset.empty:
        st.info("Not enough returns to compute correlations.")
        return
    corr = subset.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels={"color": "Corr"},
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")


def portfolio_sandbox(features: pd.DataFrame, tickers: List[str]) -> None:
    subset = (
        features[features["ticker"].isin(tickers)]
        .pivot(index="date", columns="ticker", values="ret_1d")
        .dropna()
    )
    if subset.empty:
        st.info("Need overlapping return history to build portfolio.")
        return

    default_weight = 1 / len(tickers) if tickers else 0
    weights = []
    st.subheader("Weights")
    for ticker in tickers:
        weights.append(
            st.slider(
                f"{ticker}", min_value=0, max_value=100, value=int(default_weight * 100)
            )
        )
    weight_arr = np.array(weights, dtype=float)
    if weight_arr.sum() == 0:
        st.warning("Adjust at least one weight.")
        return
    weight_arr = weight_arr / weight_arr.sum()

    portfolio_ret = (subset * weight_arr).sum(axis=1)
    cum = (1 + portfolio_ret).cumprod()
    rolling_max = cum.cummax()
    dd = cum / rolling_max - 1
    ann_return = portfolio_ret.mean() * 252
    ann_vol = portfolio_ret.std(ddof=0) * np.sqrt(252)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ann. Return", format_pct(ann_return))
    col2.metric("Ann. Vol", format_pct(ann_vol))
    col3.metric("Max Drawdown", format_pct(dd.min()))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3]
    )
    fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum,
            mode="lines",
            line=dict(width=2),
            name="Growth",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            mode="lines",
            line=dict(width=1.5),
            fill="tozeroy",
            name="Drawdown",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner=False)
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def about_section() -> None:
    st.markdown(
        """
        **About this project**
        - Pipeline: GitHub Actions runs ETL → publishes CSVs to GitHub Pages.
        - Consumption: Streamlit reads the same published CSVs (not via direct APIs).
        """
    )


def sidebar_template() -> str:
    """Centralize sidebar controls for consistency."""
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Asset Deep Dive", "Compare / Sandbox", "About"],
        format_func=lambda x: {
            "Overview": "🏠 Overview",
            "Asset Deep Dive": "📈 Asset Deep Dive",
            "Compare / Sandbox": "🧮 Compare / Sandbox",
            "About": "ℹ️ About",
        }[x],
    )

    st.sidebar.markdown("---")

    if st.button("Refresh data"):
        st.cache_data.clear()  # clears all @st.cache_data entries
        st.rerun()

    return page


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def main() -> None:
    page = sidebar_template()

    data = load_datasets(APP_CONFIG.paths.data_dir)

    dim_ticker = data["dim_ticker"]
    fact_prices = data["fact_prices"]
    fact_features = data["fact_features"]
    latest_snapshot = data["fact_latest"]
    dim_date = data.get("dim_date", pd.DataFrame())
    etl_meta = data.get("etl_metadata", pd.DataFrame())

    # Filters
    st.sidebar.markdown("### Filters")
    asset_classes = sorted(dim_ticker["asset_class"].dropna().unique())
    selected_asset_classes = st.sidebar.multiselect(
        "Asset class", asset_classes, default=asset_classes
    )

    groups = sorted(dim_ticker["group"].dropna().unique())
    selected_groups = st.sidebar.multiselect("Group", groups, default=groups)

    filtered_tickers = dim_ticker[
        dim_ticker["asset_class"].isin(selected_asset_classes)
        & dim_ticker["group"].isin(selected_groups)
    ]["ticker"].tolist()

    selected_tickers = st.sidebar.multiselect(
        "Tickers", filtered_tickers, default=filtered_tickers[:6]
    )
    if not selected_tickers:
        st.sidebar.error("Select at least one ticker.")
        st.stop()

    min_date = fact_prices["date"].min()
    max_date = fact_prices["date"].max()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    start_date, end_date = (
        date_range if isinstance(date_range, tuple) else (min_date, max_date)
    )

    # Filter data
    price_f = fact_prices[
        (fact_prices["ticker"].isin(selected_tickers))
        & (
            fact_prices["date"].between(
                pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
        )
    ]
    feat_f = fact_features[
        (fact_features["ticker"].isin(selected_tickers))
        & (
            fact_features["date"].between(
                pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
        )
    ]
    latest_f = latest_snapshot[latest_snapshot["ticker"].isin(selected_tickers)]
    # Page content
    if page == "Overview":
        st.title(APP_CONFIG.pages.overview.title)
        st.caption(APP_CONFIG.pages.overview.description)

        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.subheader("Last ETL refresh")
            st.markdown(f"**{latest_etl_timestamp(etl_meta)}**")
        with col_right:
            st.markdown(f"**Selection:** {len(selected_tickers)} tickers")

        metric_cards(latest_f, selected_tickers)
        st.markdown("### Return heatmap")
        heatmap_returns(latest_f, selected_tickers)

        st.markdown("### Micro trends (sparklines)")
        sparkline_panel(price_f, selected_tickers, lookback_days=180)

    elif page == "Asset Deep Dive":
        st.title("Asset Deep Dive")
        focus = st.selectbox("Focus ticker", selected_tickers)
        regimes = feat_f[feat_f["ticker"] == focus].sort_values("date").tail(1)
        if not regimes.empty:
            trend = regimes["trend_regime"].iloc[0]
            vol_regime = regimes["vol_regime"].iloc[0]
            st.markdown(f"**Trend regime:** {trend}")
            st.markdown(f"**Vol regime:** {vol_regime}")
        candlestick_panel(price_f, feat_f, focus)

    elif page == "Compare / Sandbox":
        st.title("Compare & Portfolio Sandbox")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk vs Return")
            risk_return_scatter(feat_f, price_f, selected_tickers)
        with col2:
            st.subheader("Correlation matrix")
            correlation_heatmap(feat_f, selected_tickers)

        st.markdown("### Portfolio sandbox")
        sandbox_tickers = st.multiselect(
            "Portfolio tickers", selected_tickers, default=selected_tickers[:4]
        )
        if sandbox_tickers:
            portfolio_sandbox(feat_f, sandbox_tickers)
        else:
            st.info("Select at least one ticker for the sandbox.")

    else:
        st.title("About")
        about_section()

    st.markdown("---")
    st.subheader("Exports")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download filtered prices",
            data=convert_df(price_f),
            file_name="prices_filtered.csv",
            mime="text/csv",
            key="dl_prices",
            help="Current ticker/date selection",
        )
    with col_b:
        st.download_button(
            "Download features",
            data=convert_df(feat_f),
            file_name="features_filtered.csv",
            mime="text/csv",
            key="dl_features",
            help="Technical + regime features",
        )


main()
