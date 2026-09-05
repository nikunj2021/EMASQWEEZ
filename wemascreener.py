"""
wemascreener.py
================
Weekly EMA alignment + proximity screener for NSE stocks (e.g. Nifty 500).

Logic
-----
1. Download weekly OHLCV data for every ticker in the universe.
2. Compute 10-week, 30-week and 40-week EMAs.
3. ABOVE-ALL condition: current weekly close > EMA10, > EMA30, and > EMA40.
4. PROXIMITY condition: the current weekly close is within 7% of *each*
   of the three EMAs (price sitting just above the moving-average
   cluster rather than extended far away from it).
5. ALIGNMENT condition: EMA10 > EMA30 > EMA40 (stacked bullish order).
6. LIQUIDITY: average weekly volume (last ~20 weeks) >= 500,000 shares.
7. Passing tickers are written to `latest_screen.md` and, if configured,
   pushed to a Google Sheet via a service account (see the "Google Sheets"
   section below).

Speed
-----
yfinance's own batched, multi-threaded downloader (`threads=True` inside a
single `yf.download()` call for a *chunk* of tickers) is used instead of
manual multiprocessing. Recent yfinance versions manage cookies/sessions
internally in a way that does not survive being forked across processes
reliably, so batched threaded downloads are the more robust way to get
concurrency here. Tickers are downloaded in chunks (default 50 per call),
and chunks are processed with a ThreadPoolExecutor so multiple batch
requests are also in flight concurrently.

Ticker universe
----------------
Reads tickers from `nifty500.csv` (a `Symbol` column, NSE symbols without
the `.NS` suffix) sitting next to this script. This is deliberately a local
file rather than a live fetch from the NSE website: NSE's site frequently
blocks requests coming from cloud/CI IP ranges (including GitHub Actions
runners), which makes a live fetch an unreliable point of failure for a
scheduled job. Keep `nifty500.csv` updated periodically (NSE publishes the
official list at https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv
which you can download locally and commit whenever the index is
reconstituted).

Google Sheets
-------------
Optional. If the environment variables `GCP_SERVICE_ACCOUNT_JSON` (the full
JSON key content of a Google service account) and `GOOGLE_SHEET_ID` (the
target spreadsheet's ID, from its URL) are both set, results are also
written to a worksheet named "Latest" in that spreadsheet, replacing its
previous contents each run. If either variable is missing, this step is
skipped with a log message — the repo file output still happens either way.
See README.md for the one-time setup (creating the service account and
sharing the sheet with it).
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("wemascreener")

# ---------------------------------------------------------------- settings

TICKER_FILE = "nifty500.csv"          # column "Symbol", no .NS suffix
OUTPUT_MD = "latest_screen.md"
OUTPUT_CSV = "latest_screen.csv"      # extra machine-readable copy

CHUNK_SIZE = 50                       # tickers per yf.download() call
MAX_WORKERS = 6                       # concurrent chunk downloads
HISTORY_PERIOD = "3y"                 # enough weekly bars for 40W EMA + lookback
MIN_BARS_REQUIRED = 45                # need 40W EMA + a bit of history

EMA_PROXIMITY_BAND = 0.07             # close must be within 7% of each EMA
VOLUME_LOOKBACK_WEEKS = 20
MIN_AVG_WEEKLY_VOLUME = 500_000       # 0.5 million shares


# ------------------------------------------------------------- data layer

def load_tickers(path: str) -> list[str]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        log.error(
            "Could not find %s. Create it with a 'Symbol' column of NSE "
            "tickers (without .NS) before running this script.",
            path,
        )
        sys.exit(1)

    if "Symbol" not in df.columns:
        log.error("%s must contain a 'Symbol' column.", path)
        sys.exit(1)

    symbols = (
        df["Symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    return [f"{s}.NS" for s in symbols]


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def download_chunk(tickers: list[str]) -> dict:
    """
    Downloads weekly data for a chunk of tickers in one batched, threaded
    yfinance call. Returns {ticker: DataFrame} for tickers that returned
    usable data; tickers with no data / errors are simply omitted (and
    logged) rather than raising.
    """
    result = {}
    try:
        data = yf.download(
            tickers=tickers,
            period=HISTORY_PERIOD,
            interval="1wk",
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # network hiccup for the whole batch
        log.warning("Batch download failed for chunk starting %s: %s", tickers[0], exc)
        return result

    # yfinance returns a single-level frame if only one ticker was requested
    if len(tickers) == 1:
        t = tickers[0]
        if not data.empty:
            result[t] = data
        return result

    for t in tickers:
        try:
            sub = data[t].dropna(how="all")
            if not sub.empty and "Close" in sub.columns:
                result[t] = sub
        except (KeyError, Exception) as exc:
            log.info("No usable data for %s (likely delisted/dead ticker): %s", t, exc)

    return result


def fetch_all(tickers: list[str]) -> dict:
    all_data = {}
    chunks = list(chunked(tickers, CHUNK_SIZE))
    log.info("Fetching %d tickers in %d chunks of up to %d...", len(tickers), len(chunks), CHUNK_SIZE)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_chunk, c): c for c in chunks}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                all_data.update(fut.result())
            except Exception as exc:
                log.warning("A chunk raised an unexpected error: %s", exc)
            if i % 2 == 0 or i == len(chunks):
                log.info("Processed %d/%d chunks...", i, len(chunks))

    return all_data


# --------------------------------------------------------------- screener

def evaluate_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """Returns a result dict if the ticker passes all conditions, else None.

    Conditions
    ----------
    1. Close above all three EMAs: close > EMA10, close > EMA30, close > EMA40.
    2. Proximity: the current weekly close is within EMA_PROXIMITY_BAND
       (7%) of *each* of the three EMAs — i.e. price is above the moving
       average cluster but not extended too far away from it.
    3. Alignment (stacked uptrend order): EMA10 > EMA30 > EMA40.
    4. Liquidity: average weekly volume over the lookback window is above
       the minimum threshold.
    """
    try:
        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < MIN_BARS_REQUIRED:
            return None

        close = df["Close"]
        volume = df["Volume"]

        ema10 = close.ewm(span=10, adjust=False).mean()
        ema30 = close.ewm(span=30, adjust=False).mean()
        ema40 = close.ewm(span=40, adjust=False).mean()

        if len(ema40) < 5:
            return None

        e10, e30, e40 = ema10.iloc[-1], ema30.iloc[-1], ema40.iloc[-1]
        last_close = close.iloc[-1]

        if any(pd.isna(v) for v in (e10, e30, e40, last_close)):
            return None
        if e10 <= 0 or e30 <= 0 or e40 <= 0:
            return None

        is_above_all = last_close > e10 and last_close > e30 and last_close > e40
        is_aligned = e10 > e30 > e40

        deviations = [abs(last_close - e) / e for e in (e10, e30, e40)]
        is_close_to_emas = all(d <= EMA_PROXIMITY_BAND for d in deviations)

        avg_vol = volume.tail(VOLUME_LOOKBACK_WEEKS).mean()
        is_liquid = avg_vol >= MIN_AVG_WEEKLY_VOLUME

        if is_above_all and is_aligned and is_close_to_emas and is_liquid:
            max_dev_pct = max(deviations) * 100
            return {
                "Symbol": ticker.replace(".NS", ""),
                "Close": round(float(last_close), 2),
                "EMA10": round(float(e10), 2),
                "EMA30": round(float(e30), 2),
                "EMA40": round(float(e40), 2),
                "MaxDeviationPct": round(float(max_dev_pct), 2),
                "AvgWeeklyVolume": int(avg_vol),
            }
        return None

    except Exception as exc:
        log.info("Skipping %s due to evaluation error: %s", ticker, exc)
        return None


def run_screen(tickers: list[str]) -> pd.DataFrame:
    raw = fetch_all(tickers)
    log.info("Got usable data for %d/%d tickers.", len(raw), len(tickers))

    passing = []
    for ticker, df in raw.items():
        res = evaluate_ticker(ticker, df)
        if res:
            passing.append(res)

    if not passing:
        return pd.DataFrame(
            columns=["Symbol", "Close", "EMA10", "EMA30", "EMA40", "MaxDeviationPct", "AvgWeeklyVolume"]
        )

    out = pd.DataFrame(passing).sort_values("MaxDeviationPct").reset_index(drop=True)
    return out


# ---------------------------------------------------------------- output

def write_outputs(df: pd.DataFrame):
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    df.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "# Weekly EMA Alignment & Proximity Screen",
        "",
        f"_Last run: {run_time}_",
        "",
        f"**{len(df)} stocks matched**: weekly close above EMA10, EMA30 and EMA40, "
        f"EMA10 > EMA30 > EMA40, close within {int(EMA_PROXIMITY_BAND * 100)}% of "
        f"all three EMAs, and >= {MIN_AVG_WEEKLY_VOLUME:,} average weekly volume.",
        "",
    ]

    if df.empty:
        lines.append("_No matches this week._")
    else:
        lines.append("| Symbol | Close | EMA10 | EMA30 | EMA40 | Max Dev % | Avg Weekly Vol |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            lines.append(
                f"| {row['Symbol']} | {row['Close']} | {row['EMA10']} | {row['EMA30']} | "
                f"{row['EMA40']} | {row['MaxDeviationPct']}% | {row['AvgWeeklyVolume']:,} |"
            )

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Wrote %s and %s", OUTPUT_MD, OUTPUT_CSV)


# -------------------------------------------------------------- gsheets

SHEET_WORKSHEET_NAME = "Latest"


def push_to_google_sheet(df: pd.DataFrame):
    """
    Writes `df` to a worksheet named SHEET_WORKSHEET_NAME in the spreadsheet
    identified by the GOOGLE_SHEET_ID env var, authenticating with the
    service account JSON in the GCP_SERVICE_ACCOUNT_JSON env var. No-ops
    (with a log message) if either variable isn't set, so this is safe to
    leave unconfigured.
    """
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")

    if not creds_json or not sheet_id:
        log.info(
            "GCP_SERVICE_ACCOUNT_JSON / GOOGLE_SHEET_ID not set — skipping "
            "Google Sheets push (repo file output already written)."
        )
        return

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        log.warning(
            "gspread/google-auth not installed but Sheets env vars are set — "
            "add them to requirements.txt to enable the Sheets push."
        )
        return

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ]
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)

        spreadsheet = client.open_by_key(sheet_id)
        try:
            ws = spreadsheet.worksheet(SHEET_WORKSHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(
                title=SHEET_WORKSHEET_NAME, rows=max(len(df) + 10, 100), cols=10
            )

        ws.clear()

        run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header_row = [f"Last run: {run_time}"]
        ws.update(values=[header_row], range_name="A1")

        if df.empty:
            ws.update(values=[["No matches this week."]], range_name="A2")
        else:
            data_rows = [df.columns.tolist()] + df.astype(str).values.tolist()
            ws.update(values=data_rows, range_name="A2")

        log.info("Pushed %d rows to Google Sheet %s (worksheet '%s').", len(df), sheet_id, SHEET_WORKSHEET_NAME)

    except Exception as exc:
        # Never let a Sheets problem fail the whole run — the repo file
        # output is already written and committed regardless.
        log.warning("Google Sheets push failed: %s", exc)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Weekly EMA alignment + proximity screener")
    parser.add_argument("--tickers-file", default=TICKER_FILE)
    args = parser.parse_args()

    start = time.time()
    tickers = load_tickers(args.tickers_file)
    df = run_screen(tickers)
    write_outputs(df)
    push_to_google_sheet(df)
    log.info("Done in %.1fs. %d matches.", time.time() - start, len(df))


if __name__ == "__main__":
    main()
