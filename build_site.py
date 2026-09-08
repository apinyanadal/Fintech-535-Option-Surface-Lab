"""
build_site.py

Orchestrates the whole pipeline (minus the LSEG pull, which already
happened in pipeline_fetch.py) and writes a self-contained static site:

    site/
      index.html
      data.js

Every (date, right) figure pair is precomputed here in Python and shipped
as JSON. The page itself is a plain HTML/JS shell -- the date dropdown,
C/P toggle, and trace show/hide checkboxes all just swap which
precomputed figure is on screen. There is no backend and no LSEG call
once this has run -- that's the only architecture that survives being
served as static files from GitHub Pages.

Usage:
    python build_site.py
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd
import plotly.io as pio

from analysis import (
    load_payload, flatten_options, attach_spot, pivot_wide, sparsity_stats,
    make_surface_figure, make_mid_vs_trade_figure, make_stock_candlestick,
)


def build_bundle(cache_file: str) -> dict:
    payload = load_payload(cache_file)
    tidy = flatten_options(payload["options"], payload["ticker"])
    wide = attach_spot(tidy, payload["stock"])
    wide = pivot_wide(wide)

    global_stats = sparsity_stats(wide)
    stock_fig = make_stock_candlestick(payload["stock"], payload["ticker"])

    dates = sorted(pd.Timestamp(d).strftime("%Y-%m-%d") for d in wide["date"].unique())
    default_asof = dates[len(dates) // 2]  # a mid-window date, not the sparsest tail

    panels = {}
    for date_str in dates:
        d = pd.Timestamp(date_str)
        day_wide = wide[wide["date"] == d]
        day_stats = sparsity_stats(day_wide) if not day_wide.empty else None
        for right in ("C", "P"):
            key = f"{date_str}|{right}"
            surface_plain = make_surface_figure(wide, d, right, show_interp_sheet=False)
            surface_sheet = make_surface_figure(wide, d, right, show_interp_sheet=True)
            compare = make_mid_vs_trade_figure(wide, d, right)
            panels[key] = {
                "surface_plain": json.loads(pio.to_json(surface_plain)),
                "surface_sheet": json.loads(pio.to_json(surface_sheet)),
                "compare": json.loads(pio.to_json(compare)),
                "day_stats": day_stats,
            }

    bundle = {
        "ticker": payload["ticker"],
        "fetched_at": payload.get("fetched_at"),
        "dates": dates,
        "default_asof": default_asof,
        "global_stats": global_stats,
        "stock_candles": json.loads(pio.to_json(stock_fig)),
        "panels": panels,
    }
    return bundle


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>UUUU Option Surface Lab</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="data.js"></script>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2.5rem 3rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: radial-gradient(circle at top left, #151a30, #0b0e1a 60%);
    color: #e8e6f0;
  }
  h1 {
    font-family: 'Georgia', serif; font-weight: 700; letter-spacing: 0.5px;
    color: #f5a623; margin: 0 0 0.25rem 0; font-size: 1.9rem;
  }
  .sub { color: #9791b3; margin: 0 0 1.5rem 0; font-size: 0.95rem; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: center;
    background: #12172a; border: 1px solid #232a45; border-radius: 12px;
    padding: 1rem 1.25rem; margin-bottom: 1.5rem;
  }
  .controls label { color: #9791b3; font-size: 0.85rem; margin-right: 0.4rem; }
  select, input[type=range] {
    background: #0b0e1a; color: #e8e6f0; border: 1px solid #232a45;
    border-radius: 6px; padding: 0.3rem 0.6rem;
  }
  .toggle { display: flex; align-items: center; gap: 0.35rem; font-size: 0.85rem; color: #cfc9e0; }
  .stats-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
  .stat-card {
    background: #12172a; border: 1px solid #232a45; border-radius: 12px;
    padding: 1rem 1.4rem; min-width: 11rem;
  }
  .stat-card .lbl { color: #9791b3; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card .val { color: #f5a623; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
  .panel {
    background: #12172a; border: 1px solid #232a45; border-radius: 12px;
    padding: 1rem; margin-bottom: 1.5rem;
  }
  .commentary {
    background: #12172a; border: 1px solid #232a45; border-radius: 12px;
    padding: 1.25rem 1.5rem; line-height: 1.6; color: #cfc9e0; font-size: 0.95rem;
  }
  .commentary strong { color: #f5a623; }
  .legend-note { color: #9791b3; font-size: 0.85rem; margin: 0.5rem 0 1rem 0; }
  .legend-note .mid { color: #f5a623; font-weight: 600; }
  .legend-note .trd { color: #ff5d73; font-weight: 600; }
  .legend-note .sheet { color: #7c5cff; font-weight: 600; }
</style>
</head>
<body>

<h1>UUUU &middot; OPTION SURFACE LAB</h1>
<p class="sub">Static snapshot &mdash; 12-week window of expired UUUU option contracts, no live LSEG connection.</p>

<div class="panel"><div id="fig_stock" style="height:360px"></div></div>

<div class="controls">
  <div><label>As-of date</label><select id="asof"></select></div>
  <div><label>Right</label>
    <select id="right"><option value="C">Calls</option><option value="P">Puts</option></select>
  </div>
  <label class="toggle"><input type="checkbox" id="show_sheet"/> Interpolated sheet</label>
</div>

<p class="legend-note">
  <span class="mid">Amber</span> = MID_PRICE, the closing NBBO midpoint &mdash; exists on far more series than trades.
  <span class="trd">Coral diamonds</span> = TRDPRC_1, an actual print &mdash; sparse by nature.
  <span class="sheet">Violet sheet</span> (off by default) is a linear interpolation across the amber cloud:
  it invents a price at every strike/expiry combination whether or not one ever printed.
</p>

<div class="stats-row" id="stats"></div>

<div class="panel"><div id="fig_surface" style="height:620px"></div></div>
<div class="panel"><div id="fig_compare" style="height:420px"></div></div>

<div class="commentary">
  <p><strong>Where is the data dense vs. empty?</strong> __COMMENT_DENSITY__</p>
  <p><strong>Why is interpolating across empty cells dangerous here?</strong> __COMMENT_INTERP__</p>
  <p><strong>Mark vs. evidence of a trade going forward:</strong> __COMMENT_MARK__</p>
</div>

<script>
const D = window.SURFACE_LAB;
const $ = (id) => document.getElementById(id);

function statCard(label, value) {
  return `<div class="stat-card"><div class="lbl">${label}</div><div class="val">${value}</div></div>`;
}

function fmtPct(x) { return (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : x.toFixed(1) + "%"; }
function fmtNum(x) { return (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : x.toFixed(3); }

function populateDateDropdown() {
  const sel = $("asof");
  sel.innerHTML = D.dates.map(d => `<option value="${d}">${d}</option>`).join("");
  sel.value = D.default_asof;
}

function render() {
  const asof = $("asof").value;
  const right = $("right").value;
  const key = asof + "|" + right;
  const panel = D.panels[key];
  const showSheet = $("show_sheet").checked;

  const g = D.global_stats;
  const day = panel ? panel.day_stats : null;
  $("stats").innerHTML = [
    statCard("Underlying", D.ticker),
    statCard("Window mid, no trade", fmtPct(g.pct_mid_no_trade)),
    statCard("Window median |mid-trade|", fmtNum(g.median_abs_diff)),
    statCard("This date, no trade", day ? fmtPct(day.pct_mid_no_trade) : "n/a"),
    statCard("This date, median |mid-trade|", day ? fmtNum(day.median_abs_diff) : "n/a"),
  ].join("");

  if (!panel) return;
  const surfaceFig = showSheet ? panel.surface_sheet : panel.surface_plain;
  Plotly.react("fig_surface", surfaceFig.data, surfaceFig.layout, {responsive: true, displaylogo: false});
  Plotly.react("fig_compare", panel.compare.data, panel.compare.layout, {responsive: true, displaylogo: false});
}

populateDateDropdown();
["asof", "right", "show_sheet"].forEach(id => $(id).addEventListener("change", render));
Plotly.newPlot("fig_stock", D.stock_candles.data, D.stock_candles.layout, {responsive: true, displaylogo: false});
render();
</script>
</body>
</html>
"""


def write_site(bundle: dict, out_dir: Path,
                comment_density: str, comment_interp: str, comment_mark: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    js = "window.SURFACE_LAB = " + json.dumps(bundle, default=str) + ";\n"
    (out_dir / "data.js").write_text(js, encoding="utf-8")

    html = (PAGE_TEMPLATE
            .replace("__COMMENT_DENSITY__", comment_density)
            .replace("__COMMENT_INTERP__", comment_interp)
            .replace("__COMMENT_MARK__", comment_mark))
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="option_pipeline_data.pkl")
    ap.add_argument("--out", default="site")
    args = ap.parse_args()

    print(f"Loading and analyzing {args.pickle}...")
    bundle = build_bundle(args.pickle)
    print(f"{bundle['ticker']}: {len(bundle['dates'])} dates, {len(bundle['panels'])} (date,right) panels.")
    print("Global sparsity stats:", bundle["global_stats"])

    # placeholder commentary -- rewrite these based on the ACTUAL numbers
    # printed above before you submit; see the chat for guidance on what
    # to say here
    write_site(
        bundle, Path(args.out),
        comment_density="We can see that the data is most dense near the spot-price plane, so where the strike is close to what the stock was trading on that given day. As we move further away in either diretion it thins out. Something worth mentioning is that the +-35% strike band used when searching the data caused some emptiness when close to expiry and high strike price, as well as far out of expiry with low strike price",
        comment_interp="Lets's say for example, that we have a real price at strike $10 and another real price at strike $12, but nothing at $11. Interpolating is drawing a straight line between the 2 known points and picking whatever value sits on that line at 11$. Nobody actually traded at $11, the chart is showing us this guess. "
                       "I also wanted to test how accurate the interpolations made are. I took prices I know are real, hid them once, and asked to guess the interpolation only using surrounding points. The results show that half the guesses were off by $0.03, and the 90th percentile off by $0.09."
                       "But the bigger issue is not the accuracy; it's that you can't tell what is interpolated by looking at the chart. This is dangerous given that you are not using data that actually happened in the market. This could lead to wrong analysis/interpretations.",
        comment_mark="I would treat the mark as the mid price. Given that this is the one that exists basically every day (93% of my rows have one). On the other hand, TRDPRICE_1 is only 71% of my rows). And we saw earlier that they are only 3.5 cents apart on a median day. So, the mid is a reasonable stand-in for what people actually pay, when we can check it.We can think of TRDPRICE_1 as our sanity check when on the days that it is available.",
    )
    print(f"Wrote {args.out}/index.html and {args.out}/data.js")


if __name__ == "__main__":
    main()
