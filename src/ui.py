"""Shared Streamlit helpers: cached loaders, colors and chart styling."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .features import SCORED_PARQUET
from .model import PriceModel, load_metrics, load_model

# Validated categorical order (blue, orange, aqua) for the three room types.
ROOM_COLORS = {
    "Entire home/apt": "#2a78d6",
    "Private room": "#eb6834",
    "Shared room": "#1baf7a",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging (below typical = blue, typical = gray, above typical = red).
VALUE_COLORS = {
    "Below typical range": "#2a78d6",
    "Typical": "#b9b8b0",
    "Above typical range": "#e34948",
    "Not modelled (price outlier)": "#898781",
}
ACCENT = "#2a78d6"
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"

pio.templates["airbnb"] = go.layout.Template(
    layout=dict(
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks=""),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks=""),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        bargap=0.25,
        colorway=list(ROOM_COLORS.values()),
    )
)
pio.templates.default = "plotly+airbnb"


@st.cache_data(show_spinner="Loading listings...")
def listings() -> pd.DataFrame:
    df = pd.read_parquet(SCORED_PARQUET)
    df["room_type"] = df["room_type"].astype(str)
    df["neighbourhood_group"] = df["neighbourhood_group"].astype(str)
    return df


@st.cache_resource(show_spinner="Loading model...")
def model() -> PriceModel:
    return load_model()


@st.cache_data
def metrics() -> dict:
    return load_metrics()


def money(x: float) -> str:
    return f"${x:,.0f}"


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="")
    has_legend = fig.layout.showlegend is not False and sum(1 for t in fig.data if t.showlegend is not False) > 1
    if has_legend and fig.layout.title.text:
        # Leave room for the legend row between the title and the plot.
        fig.update_layout(margin_t=80, title_y=0.97, title_yanchor="top")
    return fig
