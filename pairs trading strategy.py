"""
Pairs Trading — Phase 1: Data Collection (Real-Time Edition)
=============================================================
Downloads historical + latest real-time prices for Nifty 100 stocks
via yfinance. Saves clean price data for cointegration testing.

get_realtime_prices() is imported by the dashboard server (server.py)
to serve live quotes to the dashboard on demand.

Requirements:
    pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

OUTPUT_DIR = "pairs_trading_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

NIFTY100_TICKERS = [
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS",
    "SBIN.NS","INDUSINDBK.NS","BANDHANBNK.NS","FEDERALBNK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS","CHOLAFIN.NS",
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS",
    "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS",
    "POWERGRID.NS","NTPC.NS","ADANIGREEN.NS","ADANIPORTS.NS",
    "MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","TVSMOTOR.NS",
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS",
    "DABUR.NS","MARICO.NS","GODREJCP.NS","COLPAL.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS","MANKIND.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS","VEDL.NS","NMDC.NS",
    "ULTRACEMCO.NS","GRASIM.NS","SHREECEM.NS","ACC.NS","AMBUJACEM.NS","DLF.NS",
    "BHARTIARTL.NS","IDEA.NS",
    "LT.NS","ADANIENT.NS","SIEMENS.NS","ABB.NS","HAVELLS.NS",
    "TITAN.NS","TATACONSUM.NS","ZOMATO.NS",
]


# ── 1. Historical download ────────────────────────────────────────────────────

def download_prices(tickers, start, end):
    print(f"\nDownloading {len(tickers)} tickers: {start} → {end}")
    raw = yf.download(tickers=tickers, start=start, end=end,
                      auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    print(f"Raw shape: {prices.shape}")
    return prices


def clean_prices(prices, min_coverage=0.80):
    coverage     = prices.notna().mean()
    good         = coverage[coverage >= min_coverage].index.tolist()
    dropped      = set(prices.columns) - set(good)
    if dropped:
        print(f"Dropped {len(dropped)} low-coverage tickers: {', '.join(sorted(dropped))}")
    prices = prices[good].ffill(limit=5).dropna(how="all").dropna(axis=1)
    print(f"Clean: {prices.shape}  |  {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


# ── 2. Real-time prices ───────────────────────────────────────────────────────

def get_realtime_prices(tickers=None):
    """
    Fetch latest real-time price for each ticker using yfinance fast_info.
    Returns dict: { "AXISBANK.NS": { price, prev_close, change, change_pct, volume, timestamp } }

    Called by:
      - server.py  → /api/realtime  (dashboard live price feed)
      - phase3     → to append today's price to spread calculation
      - phase4     → to mark current open P&L on active positions
    """
    if tickers is None:
        tickers = NIFTY100_TICKERS

    print(f"Fetching real-time prices for {len(tickers)} tickers...")
    results = {}

    for ticker in tickers:
        try:
            info       = yf.Ticker(ticker).fast_info
            price      = round(float(info.last_price), 2)
            prev_close = round(float(info.previous_close), 2)
            change     = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

            results[ticker] = {
                "price"      : price,
                "prev_close" : prev_close,
                "change"     : change,
                "change_pct" : change_pct,
                "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            print(f"  Warning: {ticker}: {e}")
            results[ticker] = None

    ok = sum(1 for v in results.values() if v is not None)
    print(f"Fetched {ok}/{len(tickers)} successfully.")
    return results


def save_realtime_snapshot(results):
    path = os.path.join(OUTPUT_DIR, "realtime_snapshot.json")
    with open(path, "w") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "prices": results}, f, indent=2)
    print(f"Saved real-time snapshot → {path}")
    return path


# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    raw     = download_prices(NIFTY100_TICKERS, START_DATE, END_DATE)
    prices  = clean_prices(raw)
    returns = compute_log_returns(prices)

    prices.to_csv(os.path.join(OUTPUT_DIR, "prices.csv"))
    returns.to_csv(os.path.join(OUTPUT_DIR, "log_returns.csv"))
    print(f"\nSaved → {OUTPUT_DIR}/prices.csv")
    print(f"Saved → {OUTPUT_DIR}/log_returns.csv")

    print(f"\nPairs to test: {prices.shape[1]*(prices.shape[1]-1)//2:,}")

    print("\nFetching real-time snapshot...")
    rt = get_realtime_prices(prices.columns.tolist())
    save_realtime_snapshot(rt)

    print("\nPhase 1 complete.\n")
    return prices, returns


if __name__ == "__main__":
    prices, returns = main()
"""
Pairs Trading — Phase 2: Cointegration Testing (Real-Time Edition)
===================================================================
Reads cleaned prices from Phase 1, runs Engle-Granger cointegration
tests on all pairs, and outputs a ranked list of cointegrated pairs.

Also appends today's real-time price to the formation dataset before
testing, so the hedge ratios reflect the most current relationship.

Requirements:
    pip install pandas numpy statsmodels scipy tqdm
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "pairs_trading_data"
OUTPUT_DIR = "pairs_trading_data"


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_prices():
    path   = os.path.join(INPUT_DIR, "prices.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded prices: {prices.shape[0]} days × {prices.shape[1]} stocks")
    return prices


def append_realtime_row(prices):
    """
    If a real-time snapshot exists (from Phase 1), append today's prices
    as the most recent row so cointegration uses the latest data point.
    """
    snap_path = os.path.join(INPUT_DIR, "realtime_snapshot.json")
    if not os.path.exists(snap_path):
        print("No real-time snapshot found — using historical data only.")
        return prices

    with open(snap_path) as f:
        snap = json.load(f)

    rt_prices = snap.get("prices", {})
    today     = pd.Timestamp(snap["fetched_at"][:10])

    if today in prices.index:
        print(f"Today ({today.date()}) already in dataset — skipping append.")
        return prices

    row = {}
    for ticker in prices.columns:
        if rt_prices.get(ticker):
            row[ticker] = rt_prices[ticker]["price"]

    if row:
        new_row   = pd.DataFrame([row], index=[today])
        prices    = pd.concat([prices, new_row]).sort_index()
        print(f"Appended real-time row for {today.date()} ({len(row)} tickers)")

    return prices


# ── 2. Formation / trading split ──────────────────────────────────────────────

def split_windows(prices, formation_years=3):
    cutoff   = prices.index[int(len(prices) * (formation_years / 5))]
    formation = prices[prices.index <= cutoff]
    trading   = prices[prices.index >  cutoff]
    print(f"\nFormation: {formation.index[0].date()} → {formation.index[-1].date()} ({len(formation)} days)")
    print(f"Trading  : {trading.index[0].date()} → {trading.index[-1].date()} ({len(trading)} days)")
    return formation, trading


# ── 3. Cointegration test ─────────────────────────────────────────────────────

def test_cointegration(s1, s2, significance=0.05):
    score, p_value, _ = coint(s1, s2)
    if p_value >= significance:
        return None

    X           = add_constant(s2)
    result      = OLS(s1, X).fit()
    hedge_ratio = result.params.iloc[1]
    alpha       = result.params.iloc[0]
    spread      = s1 - hedge_ratio * s2 - alpha

    adf_stat    = adfuller(spread, autolag="AIC")[0]

    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    lam         = OLS(spread_diff, add_constant(spread_lag)).fit().params.iloc[1]
    half_life   = -np.log(2) / np.log(1 + lam) if lam < 0 else np.nan

    return {
        "p_value"    : round(p_value, 5),
        "hedge_ratio": round(hedge_ratio, 4),
        "alpha"      : round(alpha, 4),
        "adf_stat"   : round(adf_stat, 4),
        "half_life"  : round(half_life, 1) if not np.isnan(half_life) else np.nan,
    }


# ── 4. Test all pairs ─────────────────────────────────────────────────────────

def find_cointegrated_pairs(formation, significance=0.05, max_half_life=126, min_half_life=5):
    tickers = formation.columns.tolist()
    pairs   = list(combinations(tickers, 2))
    print(f"\nTesting {len(pairs):,} pairs (p < {significance}, HL: {min_half_life}–{max_half_life}d)...")

    results = []
    for t1, t2 in tqdm(pairs, desc="Testing"):
        s1     = formation[t1].dropna()
        s2     = formation[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 252:
            continue
        s1, s2 = s1[common], s2[common]
        r = test_cointegration(s1, s2, significance)
        if r is None:
            continue
        hl = r["half_life"]
        if np.isnan(hl) or not (min_half_life <= hl <= max_half_life):
            continue
        if r["hedge_ratio"] <= 0:
            continue
        results.append({"stock_1": t1, "stock_2": t2, **r})

    if not results:
        print("No cointegrated pairs found. Try relaxing thresholds.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(["p_value", "half_life"]).reset_index(drop=True)
    return df


# ── 5. Results & save ─────────────────────────────────────────────────────────

def summarise(df):
    if df.empty:
        return
    print(f"\n{'─'*60}")
    print(f"  Cointegrated pairs : {len(df)}")
    print(f"  Avg p-value        : {df['p_value'].mean():.4f}")
    print(f"  Avg half-life      : {df['half_life'].mean():.1f} days")
    print(f"{'─'*60}\n")
    top = df.head(15)[["stock_1","stock_2","p_value","hedge_ratio","half_life"]]
    top.index = range(1, len(top)+1)
    print(f"  {'#':<4} {'Pair':<35} {'p-value':<10} {'Hedge ratio':<14} {'Half-life (days)'}")
    print(f"  {'─'*4} {'─'*35} {'─'*10} {'─'*14} {'─'*16}")
    for i, row in top.iterrows():
        pair = f"{row['stock_1']} / {row['stock_2']}"
        print(f"  {i:<4} {pair:<35} {row['p_value']:<10.5f} {row['hedge_ratio']:<14.4f} {row['half_life']:.1f}")


def save_results(df, formation, trading):
    if df.empty:
        return
    df.to_csv(os.path.join(OUTPUT_DIR, "cointegrated_pairs.csv"), index=False)
    formation.to_csv(os.path.join(OUTPUT_DIR, "formation_prices.csv"))
    trading.to_csv(os.path.join(OUTPUT_DIR, "trading_prices.csv"))
    print(f"Saved cointegrated_pairs.csv, formation_prices.csv, trading_prices.csv")


def plot_best_spread(df, formation):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    if df.empty:
        return

    best  = df.iloc[0]
    t1, t2 = best["stock_1"], best["stock_2"]
    common = formation[t1].index.intersection(formation[t2].index)
    spread = formation[t1][common] - best["hedge_ratio"] * formation[t2][common] - best["alpha"]
    z      = (spread - spread.mean()) / spread.std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Best pair: {t1} / {t2}  (p={best['p_value']:.4f}, HL={best['half_life']:.0f}d)", fontsize=13)

    ax1 = axes[0]
    ax1.plot(formation[t1][common] / formation[t1][common].iloc[0], label=t1, lw=1.2)
    ax1.plot(formation[t2][common] / formation[t2][common].iloc[0], label=t2, lw=1.2, alpha=0.8)
    ax1.set_ylabel("Normalised price"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(z.index, z, color="steelblue", lw=1)
    ax2.axhline(0,    color="black",  lw=0.8, linestyle="--")
    ax2.axhline( 2.0, color="red",    lw=0.8, linestyle="--", label="Entry (±2σ)")
    ax2.axhline(-2.0, color="red",    lw=0.8, linestyle="--")
    ax2.axhline( 3.5, color="orange", lw=0.8, linestyle=":",  label="Stop-loss (±3.5σ)")
    ax2.axhline(-3.5, color="orange", lw=0.8, linestyle=":")
    ax2.fill_between(z.index, -2, 2, alpha=0.07, color="green")
    ax2.set_ylabel("Z-score"); ax2.legend(loc="upper right", fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "best_pair_spread.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    prices    = load_prices()
    prices    = append_realtime_row(prices)          # inject today's real-time price
    formation, trading = split_windows(prices, formation_years=3)
    pairs_df  = find_cointegrated_pairs(formation, significance=0.05,
                                        max_half_life=126, min_half_life=5)
    summarise(pairs_df)
    save_results(pairs_df, formation, trading)
    plot_best_spread(pairs_df, formation)
    print("\nPhase 2 complete.\n")
    return pairs_df, formation, trading


if __name__ == "__main__":
    pairs_df, formation, trading = main()
"""
Pairs Trading — Phase 3: Signal Generation (Real-Time Edition)
===============================================================
Generates historical signals on the trading window AND checks
live z-scores using today's real-time prices from yfinance.

The live signal check tells you RIGHT NOW whether any pair is
at an entry, exit, or stop-loss level based on current market prices.

Requirements:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "pairs_trading_data"
OUTPUT_DIR = "pairs_trading_data"

ENTRY_ZSCORE    = 2.0
EXIT_ZSCORE     = 0.0
STOPLOSS_ZSCORE = 3.5
ROLLING_WINDOW  = 30


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data():
    pairs_df  = pd.read_csv(os.path.join(INPUT_DIR, "cointegrated_pairs.csv"))
    formation = pd.read_csv(os.path.join(INPUT_DIR, "formation_prices.csv"),
                            index_col=0, parse_dates=True)
    trading   = pd.read_csv(os.path.join(INPUT_DIR, "trading_prices.csv"),
                            index_col=0, parse_dates=True)
    print(f"Loaded {len(pairs_df)} pairs | Trading: {trading.index[0].date()} → {trading.index[-1].date()}")
    return pairs_df, formation, trading


def load_realtime_snapshot():
    path = os.path.join(INPUT_DIR, "realtime_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        snap = json.load(f)
    age_mins = (datetime.now() - datetime.fromisoformat(snap["fetched_at"])).seconds // 60
    print(f"Real-time snapshot age: {age_mins} min (fetched at {snap['fetched_at'][:19]})")
    return snap


# ── 2. Spread and z-score ─────────────────────────────────────────────────────

def compute_spread(s1, s2, hedge_ratio, alpha):
    return s1 - hedge_ratio * s2 - alpha


def compute_zscore(spread, window=ROLLING_WINDOW):
    mean = spread.rolling(window=window).mean()
    std  = spread.rolling(window=window).std()
    return (spread - mean) / std


# ── 3. Signal state machine ───────────────────────────────────────────────────

def generate_signals(z):
    signals  = pd.Series(0, index=z.index, dtype=float)
    position = 0
    for i in range(1, len(z)):
        curr_z = z.iloc[i]
        if np.isnan(curr_z):
            signals.iloc[i] = 0
            continue
        if abs(curr_z) >= STOPLOSS_ZSCORE:
            position = 0
        elif position ==  1 and curr_z >= EXIT_ZSCORE:
            position = 0
        elif position == -1 and curr_z <= EXIT_ZSCORE:
            position = 0
        elif position == 0 and curr_z <= -ENTRY_ZSCORE:
            position = 1
        elif position == 0 and curr_z >=  ENTRY_ZSCORE:
            position = -1
        signals.iloc[i] = position
    return signals


# ── 4. Live z-score from real-time prices ────────────────────────────────────

def compute_live_zscore(pair_row, trading, rt_snapshot):
    """
    Compute the current live z-score for a pair using:
    - Historical spread from the trading window (to get rolling stats)
    - Today's real-time price appended as the latest data point
    """
    t1, t2       = pair_row["stock_1"], pair_row["stock_2"]
    hedge_ratio  = pair_row["hedge_ratio"]
    alpha        = pair_row["alpha"]
    half_life    = pair_row["half_life"]
    window       = max(10, min(int(half_life), 60))

    if t1 not in trading.columns or t2 not in trading.columns:
        return None

    s1 = trading[t1].dropna()
    s2 = trading[t2].dropna()
    common = s1.index.intersection(s2.index)
    s1, s2 = s1[common], s2[common]

    # Append real-time price if available
    if rt_snapshot:
        rt = rt_snapshot.get("prices", {})
        p1 = rt.get(t1, {})
        p2 = rt.get(t2, {})
        if p1 and p2:
            today = pd.Timestamp(rt_snapshot["fetched_at"][:10])
            if today not in s1.index:
                s1 = pd.concat([s1, pd.Series([p1["price"]], index=[today])])
                s2 = pd.concat([s2, pd.Series([p2["price"]], index=[today])])

    spread  = compute_spread(s1, s2, hedge_ratio, alpha)
    z       = compute_zscore(spread, window=window)
    live_z  = z.iloc[-1]

    # Determine signal
    if abs(live_z) >= STOPLOSS_ZSCORE:
        signal_label = "🔴 STOP-LOSS ZONE"
    elif abs(live_z) >= ENTRY_ZSCORE:
        signal_label = "🟢 ENTRY SIGNAL" if live_z < 0 else "🔴 SHORT SIGNAL"
    elif abs(live_z) <= EXIT_ZSCORE + 0.2:
        signal_label = "🟡 EXIT / FLAT"
    else:
        signal_label = "⚪ NO SIGNAL"

    return {
        "pair"        : f"{t1.replace('.NS','')} / {t2.replace('.NS','')}",
        "live_z"      : round(live_z, 3),
        "signal"      : signal_label,
        "s1_price"    : p1.get("price") if rt_snapshot and p1 else None,
        "s2_price"    : p2.get("price") if rt_snapshot and p2 else None,
        "s1_chg"      : p1.get("change_pct") if rt_snapshot and p1 else None,
        "s2_chg"      : p2.get("change_pct") if rt_snapshot and p2 else None,
        "half_life"   : half_life,
        "window"      : window,
        "as_of"       : rt_snapshot["fetched_at"][:19] if rt_snapshot else "historical only",
    }


# ── 5. Process all pairs ──────────────────────────────────────────────────────

def process_all_pairs(pairs_df, trading, formation, rt_snapshot, top_n=10):
    print(f"\nGenerating signals for top {top_n} pairs...")
    all_signals  = {}
    live_signals = []

    for _, row in pairs_df.head(top_n).iterrows():
        t1, t2       = row["stock_1"], row["stock_2"]
        hedge_ratio  = row["hedge_ratio"]
        alpha        = row["alpha"]
        half_life    = row["half_life"]
        pair_label   = f"{t1.replace('.NS','')} / {t2.replace('.NS','')}"
        window       = max(10, min(int(half_life), 60))

        if t1 not in trading.columns or t2 not in trading.columns:
            continue

        s1     = trading[t1].dropna()
        s2     = trading[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 60:
            continue
        s1, s2 = s1[common], s2[common]

        spread  = compute_spread(s1, s2, hedge_ratio, alpha)
        z       = compute_zscore(spread, window=window)
        signals = generate_signals(z)

        n_trades   = (signals.diff().abs() > 0).sum() // 2
        in_market  = (signals != 0).sum() / len(signals) * 100

        all_signals[pair_label] = {
            "t1": t1, "t2": t2, "hedge_ratio": hedge_ratio, "alpha": alpha,
            "spread": spread, "z_score": z, "signals": signals,
            "s1": s1, "s2": s2, "window": window,
        }

        # Live z-score
        live = compute_live_zscore(row, trading, rt_snapshot)
        if live:
            live_signals.append(live)
            live_z_str = f"  live z={live['live_z']:+.3f}  {live['signal']}"
        else:
            live_z_str = ""

        print(f"  {pair_label:<35}  trades={n_trades:>3}  in-market={in_market:>5.1f}%{live_z_str}")

    return all_signals, live_signals


# ── 6. Print live signal dashboard ───────────────────────────────────────────

def print_live_signals(live_signals):
    if not live_signals:
        return
    print(f"\n{'═'*70}")
    print("  LIVE SIGNAL DASHBOARD")
    print(f"{'═'*70}")
    print(f"  {'Pair':<35} {'Z-Score':>8}  {'S1 Price':>10}  {'S2 Price':>10}  Signal")
    print(f"  {'─'*35} {'─'*8}  {'─'*10}  {'─'*10}  {'─'*20}")
    for ls in live_signals:
        s1p = f"₹{ls['s1_price']:>8.2f}" if ls["s1_price"] else "    N/A   "
        s2p = f"₹{ls['s2_price']:>8.2f}" if ls["s2_price"] else "    N/A   "
        print(f"  {ls['pair']:<35} {ls['live_z']:>+8.3f}  {s1p}  {s2p}  {ls['signal']}")
    print(f"{'═'*70}")
    print(f"  As of: {live_signals[0]['as_of']}")
    print(f"{'═'*70}\n")

    # Save live signals JSON for dashboard
    path = os.path.join(OUTPUT_DIR, "live_signals.json")
    with open(path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "signals": live_signals}, f, indent=2)
    print(f"Saved live signals → {path}")


# ── 7. Plot and save ──────────────────────────────────────────────────────────

def plot_pair_signals(pair_label, data):
    s1, s2   = data["s1"], data["s2"]
    spread   = data["spread"]
    z        = data["z_score"]
    signals  = data["signals"]
    window   = data["window"]

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std  = spread.rolling(window=window).std()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Pairs Trading: {pair_label}", fontsize=13, fontweight="bold")

    t1l = data["t1"].replace(".NS", "")
    t2l = data["t2"].replace(".NS", "")

    axes[0].plot(s1 / s1.iloc[0], label=t1l, lw=1.3, color="#2196F3")
    axes[0].plot(s2 / s2.iloc[0], label=t2l, lw=1.3, color="#FF9800", alpha=0.85)
    axes[0].set_ylabel("Normalised Price"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25)

    axes[1].plot(spread.index, spread, lw=1, color="steelblue", label="Spread")
    axes[1].plot(rolling_mean.index, rolling_mean, lw=1.2, color="black", linestyle="--")
    axes[1].fill_between(spread.index, rolling_mean-rolling_std, rolling_mean+rolling_std,
                         alpha=0.15, color="grey")
    axes[1].set_ylabel("Spread (₹)"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25)

    axes[2].plot(z.index, z, lw=1, color="purple")
    axes[2].axhline(0, color="black", lw=0.8, linestyle="--")
    for level, col, ls in [(ENTRY_ZSCORE,"red","--"),(-ENTRY_ZSCORE,"red","--"),
                            (STOPLOSS_ZSCORE,"orange",":"),(- STOPLOSS_ZSCORE,"orange",":")]:
        axes[2].axhline(level, color=col, lw=0.9, linestyle=ls)
    axes[2].fill_between(signals.index, -5, 5, where=(signals==1),  alpha=0.12, color="green")
    axes[2].fill_between(signals.index, -5, 5, where=(signals==-1), alpha=0.12, color="red")
    axes[2].set_ylim(-5, 5); axes[2].set_ylabel("Z-score"); axes[2].grid(True, alpha=0.25)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)

    plt.tight_layout()
    safe = pair_label.replace("/","_").replace(" ","")
    path = os.path.join(OUTPUT_DIR, f"signals_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_signals(all_signals):
    rows = []
    for pair_label, data in all_signals.items():
        df = pd.DataFrame({
            "pair": pair_label, "t1": data["t1"], "t2": data["t2"],
            "hedge_ratio": data["hedge_ratio"], "alpha": data["alpha"],
            "signal": data["signals"], "z_score": data["z_score"],
            "spread": data["spread"], "s1_price": data["s1"], "s2_price": data["s2"],
        })
        rows.append(df)
    combined = pd.concat(rows).reset_index().rename(columns={"index": "Date"})
    path = os.path.join(OUTPUT_DIR, "signals.csv")
    combined.to_csv(path, index=False)
    print(f"Saved signals → {path}  ({len(combined):,} rows)")


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    pairs_df, formation, trading = load_data()
    rt_snapshot = load_realtime_snapshot()

    all_signals, live_signals = process_all_pairs(
        pairs_df, trading, formation, rt_snapshot, top_n=10)

    print_live_signals(live_signals)

    print("Plotting signal charts...")
    for pair_label, data in all_signals.items():
        plot_pair_signals(pair_label, data)
        print(f"  Saved → signals_{pair_label.replace('/','_').replace(' ','')}.png")

    save_signals(all_signals)
    print("\nPhase 3 complete.\n")
    return all_signals, live_signals


if __name__ == "__main__":
    all_signals, live_signals = main()
"""
Pairs Trading — Phase 4: Backtesting & Live P&L (Real-Time Edition)
=====================================================================
Backtests all signals from Phase 3 with realistic transaction costs.
Also computes OPEN P&L on any currently active positions using
today's real-time prices from yfinance.

Requirements:
    pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR    = "pairs_trading_data"
OUTPUT_DIR   = "pairs_trading_data"
CAPITAL      = 100_000
COST_PER_LEG = 0.0005
ANNUAL_RF    = 0.065


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_signals():
    path = os.path.join(INPUT_DIR, "signals.csv")
    df   = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"})
    print(f"Loaded signals: {len(df):,} rows, {df['pair'].nunique()} pairs")
    print(f"Date range    : {df['date'].min().date()} → {df['date'].max().date()}\n")
    return df


def load_realtime_snapshot():
    path = os.path.join(INPUT_DIR, "realtime_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_live_signals():
    path = os.path.join(INPUT_DIR, "live_signals.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("signals", [])


# ── 2. Backtest single pair ───────────────────────────────────────────────────

def backtest_pair(pair_data, capital=CAPITAL):
    df          = pair_data.sort_values("date").copy().reset_index(drop=True)
    hedge_ratio = df["hedge_ratio"].iloc[0]
    signals     = df["signal"].values
    s1          = df["s1_price"].values
    s2          = df["s2_price"].values
    n           = len(df)
    daily_pnl   = np.zeros(n)
    position    = np.zeros(n)
    trade_cost  = np.zeros(n)

    for i in range(1, n):
        prev_sig  = signals[i-1]
        curr_sig  = signals[i]
        s1_shares = capital / s1[i]
        s2_shares = (capital * hedge_ratio) / s2[i]

        if prev_sig == 1:
            pnl = s1_shares*(s1[i]-s1[i-1]) - s2_shares*(s2[i]-s2[i-1])
        elif prev_sig == -1:
            pnl = -s1_shares*(s1[i]-s1[i-1]) + s2_shares*(s2[i]-s2[i-1])
        else:
            pnl = 0.0

        daily_pnl[i] = pnl
        position[i]  = prev_sig

        if curr_sig != prev_sig:
            cost = capital * COST_PER_LEG * 2
            if curr_sig != 0 and prev_sig != 0:
                cost *= 2
            trade_cost[i] = cost

    net_pnl = daily_pnl - trade_cost
    result  = df[["date","pair","signal","z_score","spread"]].copy()
    result["gross_pnl"]  = daily_pnl
    result["trade_cost"] = trade_cost
    result["net_pnl"]    = net_pnl
    result["position"]   = position
    result["equity"]     = capital + net_pnl.cumsum()
    return result


# ── 3. Real-time open P&L ─────────────────────────────────────────────────────

def compute_open_pnl(signals_df, rt_snapshot, capital=CAPITAL):
    """
    For any pair currently in an active position (last signal != 0),
    compute the unrealised P&L using today's real-time prices.
    """
    if not rt_snapshot:
        return []

    rt      = rt_snapshot.get("prices", {})
    open_positions = []

    for pair_label in signals_df["pair"].unique():
        pair_data = signals_df[signals_df["pair"] == pair_label].sort_values("date")
        last_row  = pair_data.iloc[-1]

        if last_row["signal"] == 0:
            continue  # no open position

        t1          = last_row["t1"]
        t2          = last_row["t2"]
        hedge_ratio = last_row["hedge_ratio"]
        position    = int(last_row["signal"])  # +1 or -1

        entry_s1 = last_row["s1_price"]
        entry_s2 = last_row["s2_price"]

        rt1 = rt.get(t1)
        rt2 = rt.get(t2)
        if not rt1 or not rt2:
            continue

        live_s1 = rt1["price"]
        live_s2 = rt2["price"]

        s1_shares = capital / entry_s1
        s2_shares = (capital * hedge_ratio) / entry_s2

        if position == 1:   # long spread: long s1, short s2
            pnl = s1_shares*(live_s1-entry_s1) - s2_shares*(live_s2-entry_s2)
        else:               # short spread: short s1, long s2
            pnl = -s1_shares*(live_s1-entry_s1) + s2_shares*(live_s2-entry_s2)

        open_positions.append({
            "pair"       : pair_label,
            "position"   : "LONG spread" if position == 1 else "SHORT spread",
            "entry_s1"   : round(entry_s1, 2),
            "entry_s2"   : round(entry_s2, 2),
            "live_s1"    : round(live_s1, 2),
            "live_s2"    : round(live_s2, 2),
            "open_pnl"   : round(pnl, 2),
            "open_pnl_pct": round(pnl / capital * 100, 3),
            "as_of"      : rt_snapshot["fetched_at"][:19],
        })

    return open_positions


def print_open_positions(open_positions):
    if not open_positions:
        print("\nNo open positions currently.")
        return

    print(f"\n{'═'*75}")
    print("  OPEN POSITIONS (Real-Time P&L)")
    print(f"{'═'*75}")
    print(f"  {'Pair':<35} {'Side':<14} {'Open P&L':>10}  {'Ret%':>7}")
    print(f"  {'─'*35} {'─'*14} {'─'*10}  {'─'*7}")
    for p in open_positions:
        sign  = "+" if p["open_pnl"] >= 0 else ""
        color = "▲" if p["open_pnl"] >= 0 else "▼"
        print(f"  {p['pair']:<35} {p['position']:<14} "
              f"{sign}₹{p['open_pnl']:>8,.0f}  {sign}{p['open_pnl_pct']:>6.3f}%  {color}")
    print(f"{'═'*75}")
    total = sum(p["open_pnl"] for p in open_positions)
    sign  = "+" if total >= 0 else ""
    print(f"  Total open P&L: {sign}₹{total:,.0f}")
    print(f"  As of: {open_positions[0]['as_of']}")
    print(f"{'═'*75}\n")

    path = os.path.join(OUTPUT_DIR, "open_positions.json")
    with open(path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "positions": open_positions}, f, indent=2)
    print(f"Saved open positions → {path}")


# ── 4. Performance metrics ────────────────────────────────────────────────────

def compute_metrics(result, capital=CAPITAL):
    daily_ret = result["net_pnl"] / capital
    equity    = result["equity"]
    total_ret = (equity.iloc[-1] - capital) / capital
    n_days    = len(result)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol   = daily_ret.std() * np.sqrt(252)
    daily_rf  = ANNUAL_RF / 252
    excess    = daily_ret - daily_rf
    sharpe    = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    downside  = excess[excess < 0].std() * np.sqrt(252)
    sortino   = (ann_ret - ANNUAL_RF) / downside if downside > 0 else 0
    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max
    max_dd    = drawdown.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0

    if "position" in result.columns:
        in_pos   = result[result["position"] != 0]
        win_rate = (in_pos["net_pnl"] > 0).mean() if len(in_pos) > 0 else 0
    else:
        win_rate = (result[result["net_pnl"] != 0]["net_pnl"] > 0).mean()

    trades      = (result["signal"].diff().abs() > 0).sum() // 2 if "signal" in result.columns else 0
    gross_wins  = result[result["net_pnl"] > 0]["net_pnl"].sum()
    gross_loss  = result[result["net_pnl"] < 0]["net_pnl"].abs().sum()
    profit_fac  = gross_wins / gross_loss if gross_loss > 0 else np.inf

    return {
        "Total Return (%)"   : round(total_ret * 100, 2),
        "Ann. Return (%)"    : round(ann_ret * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio"       : round(sharpe, 3),
        "Sortino Ratio"      : round(sortino, 3),
        "Calmar Ratio"       : round(calmar, 3),
        "Max Drawdown (%)"   : round(max_dd * 100, 2),
        "Win Rate (%)"       : round(win_rate * 100, 2),
        "Profit Factor"      : round(profit_fac, 3),
        "Total Trades"       : int(trades),
        "Final Equity (₹)"   : round(equity.iloc[-1], 2),
    }


# ── 5. Tearsheet ──────────────────────────────────────────────────────────────

def plot_tearsheet(result, metrics, pair_label):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"Backtest Tearsheet — {pair_label}", fontsize=13, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    dates  = result["date"]
    equity = result["equity"]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, lw=1.5, color="#2196F3", label="Strategy equity")
    ax1.axhline(CAPITAL, color="grey", lw=0.8, linestyle="--", label="Starting capital")
    ax1.fill_between(dates, CAPITAL, equity, where=(equity>=CAPITAL), alpha=0.15, color="green")
    ax1.fill_between(dates, CAPITAL, equity, where=(equity< CAPITAL), alpha=0.15, color="red")
    ax1.set_ylabel("Portfolio Value (₹)"); ax1.set_title("Equity Curve", fontsize=10, loc="left")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20)

    ax2 = fig.add_subplot(gs[1, 0])
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    ax2.fill_between(dates, drawdown, 0, alpha=0.5, color="red")
    ax2.plot(dates, drawdown, lw=0.8, color="darkred")
    ax2.set_ylabel("Drawdown (%)"); ax2.set_title("Drawdown", fontsize=10, loc="left")
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20)

    ax3 = fig.add_subplot(gs[1, 1])
    daily_ret = result["net_pnl"]
    ax3.hist(daily_ret[daily_ret!=0], bins=40, color="#7C4DFF", alpha=0.7, edgecolor="white")
    ax3.axvline(0, color="black", lw=1)
    ax3.axvline(daily_ret.mean(), color="orange", lw=1.2, linestyle="--",
                label=f"Mean: ₹{daily_ret.mean():.0f}")
    ax3.set_xlabel("Daily P&L (₹)"); ax3.set_ylabel("Frequency")
    ax3.set_title("Daily P&L Distribution", fontsize=10, loc="left")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[2, 0])
    roll_ret    = result["net_pnl"] / CAPITAL
    roll_sharpe = roll_ret.rolling(63).apply(
        lambda x: (x.mean()-ANNUAL_RF/252)/x.std()*np.sqrt(252) if x.std()>0 else 0)
    ax4.plot(dates, roll_sharpe, lw=1.2, color="teal")
    ax4.axhline(0, color="black", lw=0.8, linestyle="--")
    ax4.axhline(1, color="green", lw=0.8, linestyle=":", label="Sharpe = 1")
    ax4.set_ylabel("Rolling Sharpe"); ax4.set_title("63-Day Rolling Sharpe", fontsize=10, loc="left")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    table = ax5.table(
        cellText  = [[k, str(v)] for k, v in metrics.items()],
        colLabels = ["Metric", "Value"],
        cellLoc="left", loc="center", colWidths=[0.65, 0.35],
    )
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0: cell.set_facecolor("#E3F2FD"); cell.set_text_props(fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#F9F9F9")
    ax5.set_title("Performance Metrics", fontsize=10, loc="left", pad=12)

    safe = pair_label.replace("/","_").replace(" ","")
    path = os.path.join(OUTPUT_DIR, f"tearsheet_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved tearsheet → {path}")


# ── 6. Portfolio ──────────────────────────────────────────────────────────────

def compute_portfolio(all_results):
    pnl_list = []
    for r in all_results:
        s = r.set_index("date")["net_pnl"].rename(r["pair"].iloc[0])
        pnl_list.append(s)
    combined  = pd.concat(pnl_list, axis=1).fillna(0)
    portfolio = combined.sum(axis=1).reset_index()
    portfolio.columns = ["date", "net_pnl"]
    total_cap = CAPITAL * len(all_results)
    portfolio["equity"] = total_cap + portfolio["net_pnl"].cumsum()
    portfolio["pair"]   = "PORTFOLIO"
    return portfolio


def plot_portfolio(portfolio):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Portfolio Equity — All Pairs Combined", fontsize=13, fontweight="bold")
    total_cap = portfolio["equity"].iloc[0] - portfolio["net_pnl"].cumsum().iloc[0]
    eq  = portfolio["equity"]

    axes[0].plot(portfolio["date"], eq, lw=1.5, color="#1565C0")
    axes[0].fill_between(portfolio["date"], total_cap, eq,
                         where=(eq>=total_cap), alpha=0.15, color="green")
    axes[0].fill_between(portfolio["date"], total_cap, eq,
                         where=(eq< total_cap), alpha=0.15, color="red")
    axes[0].axhline(total_cap, color="grey", lw=0.8, linestyle="--")
    axes[0].set_ylabel("Portfolio Value (₹)"); axes[0].grid(True, alpha=0.25)

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    axes[1].fill_between(portfolio["date"], dd, 0, alpha=0.5, color="red")
    axes[1].set_ylabel("Drawdown (%)"); axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "portfolio_equity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved portfolio chart → {path}")


# ── 7. Summary ────────────────────────────────────────────────────────────────

def print_and_save_summary(summary):
    df = pd.DataFrame(summary).sort_values("Sharpe Ratio", ascending=False)
    print(f"\n{'═'*90}")
    print("  BACKTEST SUMMARY")
    print(f"{'═'*90}")
    cols = ["pair","Total Return (%)","Ann. Return (%)","Sharpe Ratio","Max Drawdown (%)","Win Rate (%)","Total Trades"]
    print(df[cols].to_string(index=False))
    print(f"{'═'*90}")
    path = os.path.join(OUTPUT_DIR, "backtest_summary.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved summary → {path}")

    # Save JSON for dashboard
    records = df.to_dict(orient="records")
    with open(os.path.join(OUTPUT_DIR, "backtest_summary.json"), "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "results": records}, f, indent=2)


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    signals_df     = load_signals()
    rt_snapshot    = load_realtime_snapshot()
    live_signals   = load_live_signals()

    all_results = []
    summary     = []
    pairs       = signals_df["pair"].unique()

    print(f"Backtesting {len(pairs)} pairs...\n")
    for pair_label in pairs:
        pair_data = signals_df[signals_df["pair"] == pair_label].copy()
        if len(pair_data) < 60:
            continue
        result  = backtest_pair(pair_data)
        metrics = compute_metrics(result)
        print(f"  {pair_label:<38}  "
              f"Return={metrics['Ann. Return (%)']:>6.1f}%  "
              f"Sharpe={metrics['Sharpe Ratio']:>6.3f}  "
              f"MaxDD={metrics['Max Drawdown (%)']:>6.1f}%  "
              f"Trades={metrics['Total Trades']:>3}")
        plot_tearsheet(result, metrics, pair_label)
        all_results.append(result)
        summary.append({"pair": pair_label, **metrics})

    # Portfolio
    if all_results:
        print("\nComputing portfolio-level performance...")
        portfolio    = compute_portfolio(all_results)
        port_metrics = compute_metrics(portfolio, capital=CAPITAL * len(all_results))
        plot_portfolio(portfolio)
        print("\nPORTFOLIO METRICS:")
        for k, v in port_metrics.items():
            print(f"  {k:<25}: {v}")

    print_and_save_summary(summary)

    # Real-time open P&L
    print("\nChecking open positions against real-time prices...")
    open_positions = compute_open_pnl(signals_df, rt_snapshot)
    print_open_positions(open_positions)

    print("\nPhase 4 complete.\n")


if __name__ == "__main__":
    main()
"""
Pairs Trading Dashboard — Streamlit Edition
============================================
Interactive dashboard for the Nifty 100 cointegration pairs trading strategy.
Reads output files from Phase 1–4 and shows live signals, backtest results,
and open P&L in a clean, non-corporate UI.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, os
from datetime import datetime

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pairs Trading · Nifty 100",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: #0b0f1a;
    color: #e2e8f0;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #0b0f1a 0%, #111827 60%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #6366f1;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.15;
    margin: 0 0 0.5rem;
  }
  .hero-title span { color: #6366f1; }
  .hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    max-width: 620px;
    line-height: 1.6;
    margin: 0;
  }

  /* ── Stat cards ── */
  .stat-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
  .stat-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    flex: 1;
    min-width: 140px;
  }
  .stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
  }
  .stat-value.pos { color: #34d399; }
  .stat-value.neg { color: #f87171; }
  .stat-value.acc { color: #818cf8; }

  /* ── Section header ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1rem;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.75rem;
  }
  .section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    flex-shrink: 0;
  }
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    color: #94a3b8;
    text-transform: uppercase;
    margin: 0;
  }

  /* ── Signal pills ── */
  .signal-entry  { background:#064e3b; color:#34d399; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
  .signal-short  { background:#4c0519; color:#fb7185; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
  .signal-exit   { background:#1c1917; color:#fbbf24; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
  .signal-stop   { background:#450a0a; color:#fca5a5; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
  .signal-none   { background:#0f172a; color:#64748b; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }

  /* ── Tables ── */
  .stDataFrame { background: #111827 !important; }

  /* ── Timestamp badge ── */
  .ts-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 3px 10px;
    margin-left: 0.5rem;
  }

  /* ── Divider ── */
  hr { border-color: #1e293b !important; }

  /* ── Selectbox dark ── */
  .stSelectbox > div > div {
    background: #111827 !important;
    border-color: #1e293b !important;
    color: #e2e8f0 !important;
  }

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.25rem;
    border-bottom: 1px solid #1e293b;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 6px 6px 0 0;
    padding: 0.6rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #818cf8 !important;
  }

  /* ── Metric delta override ── */
  [data-testid="stMetricDelta"] { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# ─── Data loaders ────────────────────────────────────────────────────────────
DATA_DIR = "pairs_trading_data"

@st.cache_data(ttl=300)
def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data(ttl=300)
def load_csv(filename, **kwargs):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)

def safe_dt(iso_str):
    try:
        return datetime.fromisoformat(iso_str).strftime("%d %b %Y, %H:%M")
    except:
        return iso_str


# ─── Load everything ─────────────────────────────────────────────────────────
pairs_df      = load_csv("cointegrated_pairs.csv")
signals_csv   = load_csv("signals.csv", parse_dates=["Date"])
backtest_json = load_json("backtest_summary.json")
live_json     = load_json("live_signals.json")
open_pos_json = load_json("open_positions.json")
rt_json       = load_json("realtime_snapshot.json")

backtest_df   = pd.DataFrame(backtest_json["results"]) if backtest_json else None
live_signals  = live_json["signals"] if live_json else []
open_positions= open_pos_json["positions"] if open_pos_json else []


# ─── HERO ────────────────────────────────────────────────────────────────────
n_pairs  = len(pairs_df) if pairs_df is not None else "—"
as_of    = safe_dt(rt_json["fetched_at"]) if rt_json else "—"
n_live   = sum(1 for s in live_signals if "ENTRY" in s.get("signal","") or "SHORT" in s.get("signal",""))

st.markdown(f"""
<div class="hero">
  <p class="hero-eyebrow">⚡ Quantitative Strategy · Nifty 100</p>
  <h1 class="hero-title">Statistical <span>Pairs Trading</span></h1>
  <p class="hero-sub">
    Cointegration-based mean-reversion across Nifty 100 pairs.
    Engle–Granger tested · Z-score signals · Live P&L tracking.
  </p>
  <div style="margin-top:1.6rem; display:flex; gap:2rem; flex-wrap:wrap;">
    <div class="stat-card" style="flex:0 0 auto; min-width:130px;">
      <div class="stat-label">Cointegrated Pairs</div>
      <div class="stat-value acc">{n_pairs}</div>
    </div>
    <div class="stat-card" style="flex:0 0 auto; min-width:130px;">
      <div class="stat-label">Live Signals</div>
      <div class="stat-value pos">{n_live}</div>
    </div>
    <div class="stat-card" style="flex:0 0 auto; min-width:180px;">
      <div class="stat-label">Prices As Of</div>
      <div class="stat-value" style="font-size:1rem; padding-top:0.25rem; color:#94a3b8; font-family:'Space Mono',monospace;">{as_of}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡  Live Signals",
    "📈  Backtest",
    "🔬  Pair Explorer",
    "📊  Universe",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNALS
# ══════════════════════════════════════════════════════════════
with tab1:

    # ── Portfolio-level open P&L banner ──────────────────────
    if open_positions:
        total_pnl = sum(p["open_pnl"] for p in open_positions)
        sign      = "+" if total_pnl >= 0 else ""
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Open Positions", len(open_positions))
        col_b.metric("Total Open P&L", f"{sign}₹{total_pnl:,.0f}",
                     delta=f"{sign}{total_pnl/100000*100:.2f}% of capital")
        col_c.metric("As Of", open_positions[0]["as_of"])

        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Open Position Detail</p></div>', unsafe_allow_html=True)

        rows = []
        for p in open_positions:
            pnl_str  = f"+₹{p['open_pnl']:,.0f}" if p["open_pnl"] >= 0 else f"-₹{abs(p['open_pnl']):,.0f}"
            rows.append({
                "Pair"       : p["pair"],
                "Side"       : p["position"],
                "Entry S1"   : f"₹{p['entry_s1']:,.2f}",
                "Live S1"    : f"₹{p['live_s1']:,.2f}",
                "Entry S2"   : f"₹{p['entry_s2']:,.2f}",
                "Live S2"    : f"₹{p['live_s2']:,.2f}",
                "Open P&L"   : pnl_str,
                "Ret %"      : f"{p['open_pnl_pct']:+.3f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions data found. Run Phase 4 first.", icon="ℹ️")

    # ── Live signal table ─────────────────────────────────────
    st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Live Z-Score Signals</p></div>', unsafe_allow_html=True)

    if live_signals:
        def signal_color(sig):
            if "ENTRY" in sig:   return "🟢"
            if "SHORT" in sig:   return "🔴"
            if "STOP" in sig:    return "🟠"
            if "EXIT" in sig:    return "🟡"
            return "⚪"

        rows = []
        for s in live_signals:
            icon = signal_color(s.get("signal",""))
            rows.append({
                "Pair"        : s["pair"],
                "Z-Score"     : f"{s['live_z']:+.3f}",
                "Signal"      : f"{icon} {s['signal']}",
                "S1 Price"    : f"₹{s['s1_price']:,.2f}" if s.get("s1_price") else "—",
                "S1 Chg%"     : f"{s['s1_chg']:+.2f}%" if s.get("s1_chg") is not None else "—",
                "S2 Price"    : f"₹{s['s2_price']:,.2f}" if s.get("s2_price") else "—",
                "S2 Chg%"     : f"{s['s2_chg']:+.2f}%" if s.get("s2_chg") is not None else "—",
                "Half-Life"   : f"{s['half_life']:.0f}d",
            })
        sig_df = pd.DataFrame(rows)
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        # Z-score gauge chart
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Z-Score Distribution</p></div>', unsafe_allow_html=True)

        zscores = [s["live_z"] for s in live_signals]
        pairs_l = [s["pair"] for s in live_signals]
        colors  = ["#34d399" if z < -2 else "#f87171" if z > 2 else "#818cf8" if abs(z) > 1 else "#475569"
                   for z in zscores]

        fig = go.Figure(go.Bar(
            x=pairs_l, y=zscores,
            marker_color=colors,
            text=[f"{z:+.2f}" for z in zscores],
            textposition="outside",
            textfont=dict(family="Space Mono", size=10, color="#94a3b8"),
        ))
        fig.add_hline(y=2.0,  line_dash="dash", line_color="#f87171", opacity=0.6,
                      annotation_text="Entry short (2σ)", annotation_font_color="#f87171")
        fig.add_hline(y=-2.0, line_dash="dash", line_color="#34d399", opacity=0.6,
                      annotation_text="Entry long (−2σ)", annotation_font_color="#34d399")
        fig.add_hline(y=0, line_color="#334155", line_width=1)
        fig.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
            font_family="Space Grotesk", font_color="#94a3b8",
            xaxis=dict(tickangle=-30, gridcolor="#1e293b", showgrid=False),
            yaxis=dict(gridcolor="#1e293b", zeroline=False, title="Z-Score"),
            height=380, margin=dict(t=20, b=60, l=40, r=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No live signal data. Run Phase 3 to generate signals.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ══════════════════════════════════════════════════════════════
with tab2:
    if backtest_df is not None and len(backtest_df):
        # ── Portfolio-level KPIs ────────────────────────────
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Portfolio Snapshot</p></div>', unsafe_allow_html=True)

        avg_sharpe  = backtest_df["Sharpe Ratio"].mean()
        avg_ret     = backtest_df["Ann. Return (%)"].mean()
        avg_dd      = backtest_df["Max Drawdown (%)"].mean()
        avg_wr      = backtest_df["Win Rate (%)"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Ann. Return",  f"{avg_ret:+.1f}%")
        k2.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        k3.metric("Avg Max Drawdown", f"{avg_dd:.1f}%")
        k4.metric("Avg Win Rate",     f"{avg_wr:.1f}%")

        # ── Return vs Sharpe scatter ─────────────────────────
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Return vs Risk</p></div>', unsafe_allow_html=True)

        fig2 = px.scatter(
            backtest_df,
            x="Ann. Return (%)", y="Sharpe Ratio",
            size="Total Trades", color="Max Drawdown (%)",
            color_continuous_scale="RdYlGn_r",
            hover_name="pair",
            hover_data={"Win Rate (%)": True, "Profit Factor": True},
            labels={"Ann. Return (%)": "Annualised Return (%)", "Sharpe Ratio": "Sharpe Ratio"},
        )
        fig2.add_vline(x=0, line_color="#334155")
        fig2.add_hline(y=1, line_dash="dash", line_color="#818cf8", opacity=0.5,
                       annotation_text="Sharpe = 1", annotation_font_color="#818cf8")
        fig2.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            font_family="Space Grotesk", font_color="#94a3b8",
            xaxis=dict(gridcolor="#1e293b", zeroline=False),
            yaxis=dict(gridcolor="#1e293b", zeroline=False),
            coloraxis_colorbar=dict(title="Max DD%", tickfont=dict(color="#64748b")),
            height=400, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Full results table ───────────────────────────────
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">All Pair Results</p></div>', unsafe_allow_html=True)

        disp = backtest_df[[
            "pair","Ann. Return (%)","Sharpe Ratio","Sortino Ratio",
            "Max Drawdown (%)","Win Rate (%)","Profit Factor","Total Trades"
        ]].sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
        disp.index += 1
        st.dataframe(disp.style.background_gradient(
            subset=["Sharpe Ratio","Ann. Return (%)"], cmap="RdYlGn"),
            use_container_width=True)

        # ── Sharpe horizontal bar ────────────────────────────
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Sharpe Ratio by Pair</p></div>', unsafe_allow_html=True)

        bdf = disp.sort_values("Sharpe Ratio")
        bar_colors = ["#34d399" if v > 1 else "#818cf8" if v > 0 else "#f87171" for v in bdf["Sharpe Ratio"]]
        fig3 = go.Figure(go.Bar(
            x=bdf["Sharpe Ratio"], y=bdf["pair"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in bdf["Sharpe Ratio"]],
            textposition="outside",
            textfont=dict(family="Space Mono", size=9, color="#94a3b8"),
        ))
        fig3.add_vline(x=0, line_color="#334155")
        fig3.add_vline(x=1, line_dash="dash", line_color="#818cf8", opacity=0.5)
        fig3.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
            font_family="Space Grotesk", font_color="#94a3b8",
            xaxis=dict(gridcolor="#1e293b", zeroline=False),
            yaxis=dict(gridcolor="#1e293b"),
            height=max(300, len(bdf) * 30 + 60),
            margin=dict(t=10, b=10, l=10, r=60),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No backtest data found. Run Phase 4 first.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════
# TAB 3 — PAIR EXPLORER
# ══════════════════════════════════════════════════════════════
with tab3:
    if signals_csv is not None and pairs_df is not None:
        pair_list = signals_csv["pair"].unique().tolist()
        selected  = st.selectbox("Select a pair", pair_list, index=0)

        pair_signals = signals_csv[signals_csv["pair"] == selected].sort_values("Date").copy()
        pair_meta    = pairs_df[
            (pairs_df["stock_1"] + " / " + pairs_df["stock_2"]).str.replace(".NS","",regex=False)
            == selected.replace(" / ", " / ")
        ]

        # Meta strip
        if not pair_meta.empty:
            m = pair_meta.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("p-value",     f"{m['p_value']:.5f}")
            c2.metric("Hedge Ratio", f"{m['hedge_ratio']:.4f}")
            c3.metric("Alpha",       f"{m['alpha']:.4f}")
            c4.metric("Half-Life",   f"{m['half_life']:.0f} days")

        if len(pair_signals) > 0:
            # ── Spread + Z-score chart ────────────────────────
            st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Spread & Z-Score</p></div>', unsafe_allow_html=True)

            dates  = pair_signals["Date"]
            spread = pair_signals["spread"]
            z      = pair_signals["z_score"]
            sig    = pair_signals["signal"]

            fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Spread (₹)", "Z-Score"),
                                 vertical_spacing=0.08)

            # rolling stats
            w = 30
            rm = spread.rolling(w).mean()
            rs = spread.rolling(w).std()

            fig4.add_trace(go.Scatter(x=dates, y=spread, name="Spread",
                line=dict(color="#818cf8", width=1), mode="lines"), row=1, col=1)
            fig4.add_trace(go.Scatter(x=dates, y=rm, name="Mean",
                line=dict(color="#f59e0b", width=1, dash="dash"), mode="lines"), row=1, col=1)
            fig4.add_trace(go.Scatter(x=dates, y=rm+rs, name="+1σ",
                line=dict(color="#334155", width=0.5), mode="lines", showlegend=False), row=1, col=1)
            fig4.add_trace(go.Scatter(x=dates, y=rm-rs, name="−1σ",
                line=dict(color="#334155", width=0.5), mode="lines",
                fill="tonexty", fillcolor="rgba(51,65,85,0.2)", showlegend=False), row=1, col=1)

            fig4.add_trace(go.Scatter(x=dates, y=z, name="Z-Score",
                line=dict(color="#e2e8f0", width=1.2), mode="lines"), row=2, col=1)

            # shade long/short positions
            long_mask  = sig == 1
            short_mask = sig == -1
            if long_mask.any():
                fig4.add_trace(go.Scatter(
                    x=pd.concat([dates[long_mask], dates[long_mask].iloc[::-1]]),
                    y=pd.concat([z[long_mask]*0+5, z[long_mask]*0-5]),
                    fill="toself", fillcolor="rgba(52,211,153,0.08)",
                    line=dict(width=0), showlegend=False, name="Long"), row=2, col=1)
            if short_mask.any():
                fig4.add_trace(go.Scatter(
                    x=pd.concat([dates[short_mask], dates[short_mask].iloc[::-1]]),
                    y=pd.concat([z[short_mask]*0+5, z[short_mask]*0-5]),
                    fill="toself", fillcolor="rgba(248,113,113,0.08)",
                    line=dict(width=0), showlegend=False, name="Short"), row=2, col=1)

            for level, col in [(2,"#f87171"),(-2,"#34d399"),(3.5,"#fb923c"),(-3.5,"#fb923c")]:
                fig4.add_hline(y=level, line_dash="dash" if abs(level)==2 else "dot",
                               line_color=col, opacity=0.5, row=2, col=1)
            fig4.add_hline(y=0, line_color="#334155", row=2, col=1)

            fig4.update_layout(
                paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
                font_family="Space Grotesk", font_color="#94a3b8",
                height=560, margin=dict(t=40, b=20),
                legend=dict(orientation="h", y=1.02, font_color="#64748b"),
                xaxis2=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
                yaxis2=dict(gridcolor="#1e293b", range=[-5, 5]),
            )
            fig4.update_xaxes(gridcolor="#1e293b")
            st.plotly_chart(fig4, use_container_width=True)

            # ── Normalized price comparison ───────────────────
            if "s1_price" in pair_signals.columns and "s2_price" in pair_signals.columns:
                s1 = pair_signals["s1_price"]
                s2 = pair_signals["s2_price"]
                t1 = pair_signals["t1"].iloc[0].replace(".NS","")
                t2 = pair_signals["t2"].iloc[0].replace(".NS","")

                st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Normalised Prices</p></div>', unsafe_allow_html=True)

                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=dates, y=s1/s1.iloc[0], name=t1,
                    line=dict(color="#818cf8", width=1.5), mode="lines"))
                fig5.add_trace(go.Scatter(x=dates, y=s2/s2.iloc[0], name=t2,
                    line=dict(color="#f59e0b", width=1.5), mode="lines"))
                fig5.update_layout(
                    paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
                    font_family="Space Grotesk", font_color="#94a3b8",
                    height=300, margin=dict(t=10, b=10),
                    legend=dict(orientation="h", y=1.08, font_color="#64748b"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b", title="Normalised (=1 at start)"),
                )
                st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No signals data found. Run Phases 1–3 first.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════
# TAB 4 — UNIVERSE
# ══════════════════════════════════════════════════════════════
with tab4:
    if pairs_df is not None and len(pairs_df):
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Cointegrated Pairs Universe</p></div>', unsafe_allow_html=True)

        # ── p-value distribution ─────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            fig6 = go.Figure(go.Histogram(
                x=pairs_df["p_value"], nbinsx=30,
                marker_color="#6366f1", marker_line_color="#0b0f1a", marker_line_width=1,
                opacity=0.85,
            ))
            fig6.update_layout(
                title="p-value Distribution", paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
                font_family="Space Grotesk", font_color="#94a3b8",
                xaxis=dict(gridcolor="#1e293b", title="Engle-Granger p-value"),
                yaxis=dict(gridcolor="#1e293b", title="Count"),
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig6, use_container_width=True)

        with c2:
            fig7 = go.Figure(go.Histogram(
                x=pairs_df["half_life"].dropna(), nbinsx=30,
                marker_color="#34d399", marker_line_color="#0b0f1a", marker_line_width=1,
                opacity=0.85,
            ))
            fig7.update_layout(
                title="Half-Life Distribution", paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
                font_family="Space Grotesk", font_color="#94a3b8",
                xaxis=dict(gridcolor="#1e293b", title="Mean-Reversion Half-Life (days)"),
                yaxis=dict(gridcolor="#1e293b", title="Count"),
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig7, use_container_width=True)

        # ── Hedge ratio scatter ──────────────────────────────
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Pair Characteristics</p></div>', unsafe_allow_html=True)

        fig8 = px.scatter(
            pairs_df,
            x="half_life", y="hedge_ratio",
            color="p_value", size_max=14,
            color_continuous_scale="Plasma_r",
            hover_data={"stock_1": True, "stock_2": True, "p_value": ":.5f"},
            labels={"half_life": "Half-Life (days)", "hedge_ratio": "Hedge Ratio"},
        )
        fig8.update_layout(
            paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            font_family="Space Grotesk", font_color="#94a3b8",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
            coloraxis_colorbar=dict(title="p-value", tickfont=dict(color="#64748b")),
            height=380, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig8, use_container_width=True)

        # ── Top 20 pairs table ───────────────────────────────
        st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Top 20 Pairs by p-value</p></div>', unsafe_allow_html=True)
        top20 = pairs_df.head(20)[["stock_1","stock_2","p_value","hedge_ratio","alpha","half_life"]].copy()
        top20.columns = ["Stock 1","Stock 2","p-value","Hedge Ratio","Alpha","Half-Life (d)"]
        top20.index   = range(1, len(top20)+1)
        st.dataframe(top20, use_container_width=True)
    else:
        st.info("No cointegrated pairs data. Run Phase 2 first.", icon="ℹ️")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.68rem; color:#334155; padding:1rem 0;">
  Built with Streamlit · yfinance · statsmodels · plotly &nbsp;|&nbsp;
  Not investment advice &nbsp;|&nbsp;
  Strategy based on Engle-Granger cointegration (1987)
</div>
""", unsafe_allow_html=True)