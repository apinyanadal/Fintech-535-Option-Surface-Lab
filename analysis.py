"""
analysis.py

Turns the raw LSEG pickle into: a tidy long table, a wide trade-vs-mid
table, sparsity stats, and the Plotly figures for the site. No network
calls anywhere in this file -- it only ever reads option_pipeline_data.pkl.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

from ric_codec import parse_option_ric

# --- visual identity ----------------------------------------------------
# deliberately not the class-starter cyan/magenta neon -- deep indigo
# background, amber for the closing mid, coral for an actual trade print,
# violet for the (dangerous) interpolated sheet
BG_PAPER = "#0b0e1a"
BG_PLOT = "#12172a"
GRID = "#232a45"
FONT_COLOR = "#e8e6f0"
FONT_FAMILY = "Space Grotesk, Segoe UI, sans-serif"
COLOR_MID = "#f5a623"      # amber  -- MID_PRICE (the mark)
COLOR_TRADE = "#ff5d73"    # coral  -- TRDPRC_1 (an actual print)
COLOR_SHEET = "#7c5cff"    # violet -- interpolated surface (use with caution)

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG_PLOT,
    font=dict(color=FONT_COLOR, family=FONT_FAMILY),
    margin=dict(l=10, r=10, t=40, b=10),
)


# --- load -----------------------------------------------------------------

def load_payload(cache_file: str = "option_pipeline_data.pkl") -> dict:
    path = Path(cache_file)
    if not path.exists():
        raise FileNotFoundError(
            f"{cache_file} not found. Run pipeline_fetch.py locally with "
            f"LSEG Workspace open first."
        )
    with path.open("rb") as f:
        return pickle.load(f)


# --- flatten wide LSEG frame -> tidy long table ----------------------------

def flatten_options(df_options: pd.DataFrame, root: str) -> pd.DataFrame:
    """
    df_options columns are a (ric, field) MultiIndex, index is date.
    Returns one row per (date, ric, field) with the parsed contract
    identifiers attached. Rows for RICs that don't parse (never-existed
    synthetic candidates) or fields we don't care about are dropped.
    """
    if df_options is None or df_options.empty:
        raise ValueError("flatten_options: options frame is empty -- nothing to analyze.")

    frame = df_options.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)

    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(
            f"flatten_options: expected a (ric, field) MultiIndex, got "
            f"flat columns {list(frame.columns)[:5]}..."
        )

    # figure out which level holds field names vs RIC strings
    level0_vals = set(map(str, frame.columns.get_level_values(0)))
    field_level = 0 if level0_vals & {"TRDPRC_1", "MID_PRICE"} else 1
    ric_level = 1 - field_level

    rows = []
    for col in frame.columns:
        ric = str(col[ric_level])
        field = str(col[field_level]).upper()
        if field not in ("TRDPRC_1", "MID_PRICE"):
            continue
        parsed = parse_option_ric(ric, root=root)
        if parsed is None:
            continue  # synthetic candidate that never existed, or malformed
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        for ts, val in series.items():
            rows.append({
                "date": pd.Timestamp(ts).normalize(),
                "ric": ric,
                "field": field,
                "value": float(val),
                **{k: v for k, v in parsed.items() if k != "ric"},
            })

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        raise ValueError("flatten_options: no rows survived parsing -- check root/RIC scheme.")
    tidy["expiry"] = pd.to_datetime(tidy["expiry"])
    tidy["dte"] = (tidy["expiry"] - tidy["date"]).dt.days
    tidy = tidy[tidy["dte"] >= 0].copy()  # drop anything expiry-before-asof (shouldn't happen, but be sure)
    return tidy


# --- attach spot / moneyness -----------------------------------------------

def attach_spot(tidy: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
    stock = df_stock.copy()
    if not isinstance(stock.index, pd.DatetimeIndex):
        stock.index = pd.to_datetime(stock.index)
    stock.index = stock.index.normalize()
    if "TRDPRC_1" not in stock.columns:
        raise ValueError(f"attach_spot: stock frame missing TRDPRC_1, has {list(stock.columns)}")

    spot_map = pd.to_numeric(stock["TRDPRC_1"], errors="coerce").to_dict()
    out = tidy.copy()
    out["spot"] = out["date"].map(spot_map)
    missing = out["spot"].isna()
    if missing.any():
        # forward-fill would hide a real gap -- drop instead, and say so
        n = int(missing.sum())
        print(f"attach_spot: dropping {n} option rows with no matching stock print on their date.")
        out = out.loc[~missing].copy()
    out["moneyness"] = out["strike"] / out["spot"]
    return out


# --- wide trade-vs-mid table + sparsity stats -------------------------------

def pivot_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ["date", "ric", "right", "strike", "expiry", "dte", "spot", "moneyness"]
    idx_cols = [c for c in idx_cols if c in tidy.columns]
    wide = (
        tidy.pivot_table(index=idx_cols, columns="field", values="value", aggfunc="last")
        .reset_index()
    )
    wide.columns.name = None
    for col in ("TRDPRC_1", "MID_PRICE"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide["has_trade"] = wide["TRDPRC_1"].notna()
    wide["has_mid"] = wide["MID_PRICE"].notna()
    wide["abs_diff"] = (wide["MID_PRICE"] - wide["TRDPRC_1"]).abs()
    return wide


def sparsity_stats(wide: pd.DataFrame) -> dict:
    """
    pct_mid_no_trade: of series-dates with a MID_PRICE, what % have no
        matching trade that day.
    median_abs_diff: median |MID_PRICE - TRDPRC_1| on rows where both exist.
    """
    listed_with_mid = wide[wide["has_mid"]]
    if len(listed_with_mid) == 0:
        pct_mid_no_trade = float("nan")
    else:
        pct_mid_no_trade = 100.0 * (listed_with_mid["has_trade"] == False).mean()

    both = wide[wide["has_mid"] & wide["has_trade"]]
    median_abs_diff = float(both["abs_diff"].median()) if len(both) else float("nan")

    return {
        "n_rows": int(len(wide)),
        "n_with_mid": int(wide["has_mid"].sum()),
        "n_with_trade": int(wide["has_trade"].sum()),
        "n_with_both": int(len(both)),
        "pct_mid_no_trade": pct_mid_no_trade,
        "median_abs_diff": median_abs_diff,
    }


# --- figures ----------------------------------------------------------------

def make_surface_figure(wide: pd.DataFrame, asof_date, right: str,
                         show_interp_sheet: bool = False) -> go.Figure:
    """
    Scatter3d of real (strike, dte, price) points for one as-of date and
    one right -- MID_PRICE and TRDPRC_1 as two separately-colored traces
    so the holes in TRDPRC_1 are visually obvious against the denser
    MID_PRICE cloud. An optional interpolated Surface trace can be laid
    under it for the "why is this dangerous" discussion -- off by default.
    """
    day = wide[(wide["date"] == pd.Timestamp(asof_date)) & (wide["right"] == right)]
    fig = go.Figure()

    if day.empty:
        fig.update_layout(**LAYOUT_BASE, title=f"No data for {right} on {asof_date}")
        return fig

    if show_interp_sheet:
        cloud = day.dropna(subset=["strike", "dte", "MID_PRICE"])
        if len(cloud) >= 8 and cloud["strike"].nunique() > 1 and cloud["dte"].nunique() > 1:
            xi = np.linspace(cloud["strike"].min(), cloud["strike"].max(), 40)
            yi = np.linspace(cloud["dte"].min(), cloud["dte"].max(), 25)
            XX, YY = np.meshgrid(xi, yi)
            try:
                ZZ = griddata(
                    (cloud["strike"], cloud["dte"]), cloud["MID_PRICE"], (XX, YY), method="linear"
                )
                fig.add_trace(go.Surface(
                    x=xi, y=yi, z=ZZ, opacity=0.35, showscale=False,
                    colorscale=[[0, COLOR_SHEET], [1, COLOR_SHEET]],
                    name="Interpolated sheet (MID)",
                ))
            except Exception:
                pass  # degenerate cloud -- skip the sheet, keep the real points

    mid_pts = day.dropna(subset=["MID_PRICE"])
    fig.add_trace(go.Scatter3d(
        x=mid_pts["strike"], y=mid_pts["dte"], z=mid_pts["MID_PRICE"],
        mode="markers", name="MID_PRICE",
        marker=dict(size=4, color=COLOR_MID, opacity=0.85),
    ))

    trd_pts = day.dropna(subset=["TRDPRC_1"])
    fig.add_trace(go.Scatter3d(
        x=trd_pts["strike"], y=trd_pts["dte"], z=trd_pts["TRDPRC_1"],
        mode="markers", name="TRDPRC_1",
        marker=dict(size=5, color=COLOR_TRADE, opacity=0.95, symbol="diamond"),
    ))
    # spot-price reference plane -- makes "near-the-money" visible at a
    # glance instead of requiring a cross-reference to the candlestick
    spot_vals = day["spot"].dropna()
    if not spot_vals.empty:
        spot_val = float(spot_vals.iloc[0])
        dte_min, dte_max = day["dte"].min(), day["dte"].max()
        price_vals = pd.concat([day["MID_PRICE"], day["TRDPRC_1"]]).dropna()
        z_top = float(price_vals.max()) * 1.1 if not price_vals.empty else 1.0
        fig.add_trace(go.Mesh3d(
            x=[spot_val] * 4,
            y=[dte_min, dte_max, dte_max, dte_min],
            z=[0, 0, z_top, z_top],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#ffffff", opacity=0.12,
            name=f"Spot \u2248 ${spot_val:.2f}",
            showlegend=True, hoverinfo="skip",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"{right} surface -- {pd.Timestamp(asof_date).date()}",
        scene=dict(
            xaxis=dict(title="Strike ($)", gridcolor=GRID, backgroundcolor=BG_PLOT),
            yaxis=dict(title="Days to expiry", gridcolor=GRID, backgroundcolor=BG_PLOT),
            zaxis=dict(title="Option price ($)", gridcolor=GRID, backgroundcolor=BG_PLOT),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6)),
        ),
        legend=dict(orientation="h", y=0),
    )
    return fig


def make_mid_vs_trade_figure(wide: pd.DataFrame, asof_date, right: str) -> go.Figure:
    """2D overlay: MID_PRICE (continuous-ish) vs TRDPRC_1 (sparse dots) across strike, one expiry slice."""
    day = wide[(wide["date"] == pd.Timestamp(asof_date)) & (wide["right"] == right)].sort_values("strike")
    fig = go.Figure()
    if day.empty:
        fig.update_layout(**LAYOUT_BASE, title="No data")
        return fig

    fig.add_trace(go.Scatter(
        x=day["strike"], y=day["MID_PRICE"], mode="lines+markers",
        name="MID_PRICE", line=dict(color=COLOR_MID, width=3),
        marker=dict(size=6),
    ))
    trd = day.dropna(subset=["TRDPRC_1"])
    fig.add_trace(go.Scatter(
        x=trd["strike"], y=trd["TRDPRC_1"], mode="markers",
        name="TRDPRC_1", marker=dict(size=10, color=COLOR_TRADE, symbol="diamond"),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"MID_PRICE vs TRDPRC_1 -- {right} -- {pd.Timestamp(asof_date).date()}",
        xaxis=dict(title="Strike", gridcolor=GRID),
        yaxis=dict(title="Price", gridcolor=GRID),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def make_stock_candlestick(df_stock: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Candlestick of the underlying over the whole pull window -- shown once,
    not per as-of-date, since it's context for the whole dataset rather
    than something that changes with the date/right toggle.
    """
    stock = df_stock.copy()
    if not isinstance(stock.index, pd.DatetimeIndex):
        stock.index = pd.to_datetime(stock.index)

    fig = go.Figure(data=[go.Candlestick(
        x=stock.index,
        open=stock["OPEN_PRC"], high=stock["HIGH_1"],
        low=stock["LOW_1"], close=stock["TRDPRC_1"],
        increasing_line_color=COLOR_MID, increasing_fillcolor=COLOR_MID,
        decreasing_line_color=COLOR_TRADE, decreasing_fillcolor=COLOR_TRADE,
        name=ticker,
    )])
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"{ticker} -- underlying, full pull window",
        xaxis=dict(title="Date", gridcolor=GRID, rangeslider=dict(visible=False)),
        yaxis=dict(title="Price ($)", gridcolor=GRID),
        showlegend=False,
    )
    return fig
