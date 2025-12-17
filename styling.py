from __future__ import annotations

from typing import Dict

import streamlit as st

DARK_THEME: Dict[str, str] = {
    "bg": "#0f172a",
    "surface": "#111827",
    "card": "#111827",
    "muted": "#94a3b8",
    "text": "#e2e8f0",
    "accent": "#22d3ee",
    "accent_muted": "#0ea5e9",
    "positive": "#34d399",
    "negative": "#f87171",
}

LIGHT_THEME: Dict[str, str] = {
    "bg": "#f8fafc",
    "surface": "#ffffff",
    "card": "#ffffff",
    "muted": "#475569",
    "text": "#0f172a",
    "accent": "#0ea5e9",
    "accent_muted": "#0284c7",
    "positive": "#16a34a",
    "negative": "#ef4444",
}


def inject_css(theme: Dict[str, str]) -> None:
    """Inject global styles for a product-like feel."""
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {theme["bg"]};
                --surface: {theme["surface"]};
                --card: {theme["card"]};
                --muted: {theme["muted"]};
                --text: {theme["text"]};
                --accent: {theme["accent"]};
                --accent-muted: {theme["accent_muted"]};
                --positive: {theme["positive"]};
                --negative: {theme["negative"]};
            }}
            .stApp {{
                background: radial-gradient(140% 120% at 0% 0%, rgba(34,211,238,0.07), transparent 40%),
                            radial-gradient(160% 140% at 100% 20%, rgba(14,165,233,0.08), transparent 42%),
                            var(--bg);
                color: var(--text);
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1400px;
            }}
            h1, h2, h3, h4 {{
                color: var(--text);
                letter-spacing: -0.02em;
            }}
            .metric-card {{
                background: linear-gradient(135deg, rgba(34,211,238,0.12), transparent 40%), var(--card);
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 16px;
                padding: 12px 16px;
                box-shadow: 0 12px 40px rgba(0,0,0,0.18);
                backdrop-filter: blur(8px);
            }}
            .pill {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 12px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.2);
                color: var(--text);
                background: rgba(255,255,255,0.04);
                font-size: 0.9rem;
            }}
            .sidebar .sidebar-content {{
                background: var(--surface);
            }}
            .stButton > button {{
                background: linear-gradient(90deg, var(--accent), var(--accent-muted));
                color: white;
                border-radius: 10px;
                border: none;
                padding: 0.4rem 1rem;
                font-weight: 600;
            }}
            .card-title {{
                color: var(--muted);
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
            }}
            .download-btn button {{
                width: 100%;
                border: 1px solid rgba(148,163,184,0.4) !important;
                background: var(--surface) !important;
                color: var(--text) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

