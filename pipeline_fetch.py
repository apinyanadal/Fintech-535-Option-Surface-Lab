"""
pipeline_fetch.py

Run this LOCALLY, with LSEG Workspace open and logged in. This is the only
script in the project that touches the network -- everything downstream
(parsing, reshaping, plotting, site export) reads the pickle this writes
and never calls LSEG again.

    python pipeline_fetch.py

Produces option_pipeline_data.pkl in the current directory.

Fixes vs. the original class handout starter:
  - fetches MID_PRICE instead of SETTLE. SETTLE returns "universe does not
    support" on this RIC space -- it is dead on arrival. MID_PRICE is the
    closing NBBO midpoint and is the field the assignment actually wants.
  - bands strikes per-expiry instead of taking one high/low across the
    whole window and generating every strike for every expiry. The
    un-banded version requests far more candidate RICs than can possibly
    exist (a name's price range drifts over 12 weeks, but any single
    expiry only ever had strikes near where the stock was trading around
    that expiry) -- banding cuts the request count a lot without losing
    real contracts.
"""

from __future__ import annotations

import os
import pickle
import datetime
import warnings

import pandas as pd
import numpy as np
import lseg.data as ld

from ric_codec import build_option_ric

warnings.filterwarnings("ignore", category=FutureWarning, module="lseg.data")

TICKER_STOCK = "UUUU.K"
TICKER_ROOT = "UUUU"
WEEKS_BACK = 12
STRIKE_STEP = 0.50
BATCH_SIZE = 25
CACHE_FILE = "option_pipeline_data.pkl"

# how far above/below the spot price observed *near a given expiry* to
# generate candidate strikes for that expiry -- this is the banding fix
STRIKE_BAND_PCT = 0.35  # +/- 35% around the local spot


def check_for_split(df_stock: pd.DataFrame) -> None:
    """
    Cheap split guard: a split shows up as an overnight price jump with no
    matching jump in the rest of the window. This is not a real corporate-
    actions lookup -- it is a sanity check, not proof. Print a warning
    rather than raising, since a real dividend/gap could also trip it.
    """
    close = df_stock["TRDPRC_1"].dropna()
    if len(close) < 3:
        return
    pct_change = close.pct_change().abs()
    jumps = pct_change[pct_change > 0.30]  # >30% overnight move is suspicious
    if not jumps.empty:
        print("WARNING: possible split or corporate action detected on:")
        for dt_, pct in jumps.items():
            print(f"    {dt_.date()}  {pct:.0%} overnight move")
        print("    Check this before trusting the synthesized RICs below "
              "these dates -- a split will make them silently miss the "
              "adjusted contracts.")


def local_spot_near(df_stock: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """Nearest available close to a given date, for per-expiry strike banding."""
    idx = df_stock.index.get_indexer([target_date], method="nearest")[0]
    return float(df_stock["TRDPRC_1"].iloc[idx])


def build_candidate_rics(df_stock: pd.DataFrame, start_str: str, end_str: str) -> list[str]:
    """Generate candidate calls+puts for every Friday in the window, with
    strikes banded around that Friday's approximate spot rather than the
    whole window's high/low."""
    fridays = pd.date_range(start=start_str, end=end_str, freq="W-FRI")
    candidates = []
    for expiry in fridays:
        spot_near_expiry = local_spot_near(df_stock, expiry)
        lo = spot_near_expiry * (1 - STRIKE_BAND_PCT)
        hi = spot_near_expiry * (1 + STRIKE_BAND_PCT)
        lo = max(STRIKE_STEP, np.floor(lo / STRIKE_STEP) * STRIKE_STEP)
        hi = np.ceil(hi / STRIKE_STEP) * STRIKE_STEP
        strikes = np.arange(lo, hi + STRIKE_STEP, STRIKE_STEP)

        year2 = int(expiry.strftime("%y"))
        for strike in strikes:
            candidates.append(build_option_ric(
                TICKER_ROOT, month=expiry.month, day=expiry.day,
                year2=year2, strike=float(strike), right="C"))
            candidates.append(build_option_ric(
                TICKER_ROOT, month=expiry.month, day=expiry.day,
                year2=year2, strike=float(strike), right="P"))
    return candidates


def fetch_options_history(candidate_rics: list[str], start_str: str, end_str: str) -> pd.DataFrame:
    fields = ["TRDPRC_1", "MID_PRICE"]  # <-- the fix: not SETTLE
    batches = [candidate_rics[i:i + BATCH_SIZE] for i in range(0, len(candidate_rics), BATCH_SIZE)]
    history_frames = []

    for batch in batches:
        try:
            df_batch = ld.get_history(universe=batch, fields=fields,
                                       start=start_str, end=end_str, interval="daily")
            if df_batch is not None and not df_batch.empty:
                df_clean = df_batch.dropna(how="all", axis=1)
                if not df_clean.empty:
                    history_frames.append(df_clean)
        except Exception:
            for single_ric in batch:
                try:
                    df_single = ld.get_history(universe=[single_ric], fields=fields,
                                                start=start_str, end=end_str, interval="daily")
                    if df_single is not None and not df_single.empty and not df_single.dropna(how="all").empty:
                        history_frames.append(df_single)
                except Exception:
                    continue

    if not history_frames:
        return pd.DataFrame()
    df_options = pd.concat(history_frames, axis=1)
    df_options = df_options.loc[:, ~df_options.columns.duplicated()]
    return df_options


def main():
    if os.path.exists(CACHE_FILE):
        print(f"{CACHE_FILE} already exists -- delete it first if you want a fresh pull.")
        return

    print("Opening LSEG session (Workspace must be running and logged in)...")
    ld.open_session()
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(weeks=WEEKS_BACK)
        start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        print(f"Pulling {TICKER_STOCK} OHLC {start_str} -> {end_str}...")
        df_stock = ld.get_history(
            universe=[TICKER_STOCK],
            fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
            start=start_str, end=end_str, interval="daily",
        )
        check_for_split(df_stock)

        candidate_rics = build_candidate_rics(df_stock, start_str, end_str)
        print(f"Generated {len(candidate_rics)} candidate option RICs "
              f"(banded per-expiry, not whole-window high/low).")

        print("Pulling options history (TRDPRC_1, MID_PRICE) in batches...")
        df_options = fetch_options_history(candidate_rics, start_str, end_str)
        print(f"Got data back for {df_options.shape[1] if not df_options.empty else 0} RIC/field columns.")

    finally:
        ld.close_session()

    payload = {
        "stock": df_stock,
        "options": df_options,
        "ticker": TICKER_ROOT,
        "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"Wrote {CACHE_FILE}.")


if __name__ == "__main__":
    main()
