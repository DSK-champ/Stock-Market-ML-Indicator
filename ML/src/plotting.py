"""Modular plotting library for data quality reports and exploratory data analysis.

Generates high-resolution publication-quality figures with clear titles, units,
and split shading. Saves figures directly to results/figures/ while returning
figure handles for inline rendering inside Jupyter notebooks.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

from src import config

# Set default styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 120

FIGURES_DIR = config.FIGURES_DIR


# =====================================================================
# NOTEBOOK 02 PLOTS (FULL PERIOD OVERVIEW & DATA QUALITY)
# =====================================================================

def plot_full_period_price(
    df: pd.DataFrame,
    log_scale: bool = False,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot closing price over the full 3-year period with shaded Train/Val/Test splits."""
    fig, ax = plt.subplots(figsize=(14, 5))
    close = df["close"]

    ax.plot(close.index, close.values, color="#1E88E5", linewidth=1.1, label="BTCUSDT Close (5m)")

    t_train_start = config.SPLIT_BOUNDARIES["train"]["start"]
    t_train_end = config.SPLIT_BOUNDARIES["train"]["end"]
    t_val_start = config.SPLIT_BOUNDARIES["val"]["start"]
    t_val_end = config.SPLIT_BOUNDARIES["val"]["end"]
    t_test_start = config.SPLIT_BOUNDARIES["test"]["start"]
    t_test_end = config.SPLIT_BOUNDARIES["test"]["end"]

    # Shading
    ax.axvspan(t_train_start, t_train_end, color="#4CAF50", alpha=0.12, label="Train Split (70%, 220,953 candles)")
    ax.axvspan(t_val_start, t_val_end, color="#FF9800", alpha=0.12, label="Validation Split (15%, 47,347 candles)")
    ax.axvspan(t_test_start, t_test_end, color="#E91E63", alpha=0.12, label="Test Split (Locked, 49,940 candles)")

    # Boundary lines
    ax.axvline(t_train_end, color="#388E3C", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(t_val_end, color="#F57C00", linestyle="--", linewidth=1.2, alpha=0.8)

    scale_text = "Log Scale" if log_scale else "Linear Scale"
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Price (USDT, Log Scale)")
    else:
        ax.set_ylabel("Price (USDT)")

    ax.set_title(f"BTCUSDT 5-Minute Price History — Full Historical Period ({scale_text})\n[Overview Only — No Modeling Decisions Derived From Test Period]")
    ax.set_xlabel("Date (UTC)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_tail_zoom(
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot zoomed-in histograms of the extreme negative and positive return tails."""
    s = returns_series.dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    q01 = float(s.quantile(0.01))
    q99 = float(s.quantile(0.99))
    
    left_tail = s[s <= q01]
    right_tail = s[s >= q99]
    
    ax1.hist(left_tail * 100.0, bins=40, color="#C62828", alpha=0.75, edgecolor="none")
    ax1.set_title(f"Left Tail Zoom (Lowest 1%, returns <= {q01*100:.2f}%)")
    ax1.set_xlabel("Return (%)")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=":", alpha=0.4)
    
    ax2.hist(right_tail * 100.0, bins=40, color="#2E7D32", alpha=0.75, edgecolor="none")
    ax2.set_title(f"Right Tail Zoom (Highest 1%, returns >= {q99*100:.2f}%)")
    ax2.set_xlabel("Return (%)")
    ax2.set_ylabel("Count")
    ax2.grid(True, linestyle=":", alpha=0.4)
    
    fig.suptitle("Zoom on Extreme 1% Tails of Train 5m Returns (N=220,953)", fontsize=12, y=0.99)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig

# =====================================================================
# 28-BAR LABEL DISTRIBUTION PLOTS
# =====================================================================

def plot_label_counts_all(
    balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ONE stacked bar per label column (28 bars): Up green, Down red, Neutral grey.
    
    Annotates integer counts directly on the bar segments.
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    labels = balance_df["label_col"].tolist()
    up_cnt = balance_df["n_up"].to_numpy()
    dn_cnt = balance_df["n_down"].to_numpy()
    neu_cnt = balance_df["n_neutral"].to_numpy()
    
    x = np.arange(len(labels))
    w = 0.75

    # Stack: Down at bottom, Neutral in middle, Up on top
    p_dn = ax.bar(x, dn_cnt, w, label="Down (-1)", color="#C62828", alpha=0.9)
    p_neu = ax.bar(x, neu_cnt, w, bottom=dn_cnt, label="Neutral (0)", color="#9E9E9E", alpha=0.8)
    p_up = ax.bar(x, up_cnt, w, bottom=dn_cnt + neu_cnt, label="Up (+1)", color="#2E7D32", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("target_", "") for l in labels], rotation=65, ha="right", fontsize=9)
    ax.set_ylabel("Candle Count")
    ax.set_title("Label Class Counts Across All 28 Target Columns (Train Only, N=220,953)", fontsize=13, pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate counts inside bars
    for i in range(len(labels)):
        # Down
        if dn_cnt[i] > 10000:
            ax.text(i, dn_cnt[i] / 2, f"{dn_cnt[i]:,}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        # Neutral
        if neu_cnt[i] > 20000:
            ax.text(i, dn_cnt[i] + neu_cnt[i] / 2, f"{neu_cnt[i]:,}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        # Up
        if up_cnt[i] > 10000:
            ax.text(i, dn_cnt[i] + neu_cnt[i] + up_cnt[i] / 2, f"{up_cnt[i]:,}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_label_percent_all(
    balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ONE stacked bar per label column (28 bars) normalized to 100%.
    
    Annotates percentages directly on the segments.
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    labels = balance_df["label_col"].tolist()
    up_pct = balance_df["pct_up"].to_numpy()
    dn_pct = balance_df["pct_down"].to_numpy()
    neu_pct = balance_df["pct_neutral"].to_numpy()
    
    x = np.arange(len(labels))
    w = 0.75

    p_dn = ax.bar(x, dn_pct, w, label="Down (-1)", color="#C62828", alpha=0.9)
    p_neu = ax.bar(x, neu_pct, w, bottom=dn_pct, label="Neutral (0)", color="#9E9E9E", alpha=0.8)
    p_up = ax.bar(x, up_pct, w, bottom=dn_pct + neu_pct, label="Up (+1)", color="#2E7D32", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("target_", "") for l in labels], rotation=65, ha="right", fontsize=9)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Label Class Proportions (100% Stacked) Across All 28 Rules (Train Only)", fontsize=13, pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for i in range(len(labels)):
        if dn_pct[i] > 5.0:
            ax.text(i, dn_pct[i] / 2, f"{dn_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        if neu_pct[i] > 10.0:
            ax.text(i, dn_pct[i] + neu_pct[i] / 2, f"{neu_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        if up_pct[i] > 5.0:
            ax.text(i, dn_pct[i] + neu_pct[i] + up_pct[i] / 2, f"{up_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_label_by_horizon(
    balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 4 panels (one per horizon N in {2,3,6,12}), each showing the 7 rules."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    axes = axes.flatten()
    horizons = [2, 3, 6, 12]
    
    for idx, N in enumerate(horizons):
        ax = axes[idx]
        sub = balance_df[balance_df["horizon"] == N].copy()
        rules = sub["theta_rule"].tolist()
        up_pct = sub["pct_up"].to_numpy()
        dn_pct = sub["pct_down"].to_numpy()
        neu_pct = sub["pct_neutral"].to_numpy()
        
        x = np.arange(len(rules))
        w = 0.65

        ax.bar(x, dn_pct, w, label="Down (-1)", color="#C62828", alpha=0.9)
        ax.bar(x, neu_pct, w, bottom=dn_pct, label="Neutral (0)", color="#9E9E9E", alpha=0.8)
        ax.bar(x, up_pct, w, bottom=dn_pct + neu_pct, label="Up (+1)", color="#2E7D32", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(rules, fontsize=9)
        ax.set_title(f"Horizon N={N} ({N*5}m)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Share (%)" if idx % 2 == 0 else "")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for i in range(len(rules)):
            if dn_pct[i] > 5.0:
                ax.text(i, dn_pct[i] / 2, f"{dn_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
            if neu_pct[i] > 10.0:
                ax.text(i, dn_pct[i] + neu_pct[i] / 2, f"{neu_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
            if up_pct[i] > 5.0:
                ax.text(i, dn_pct[i] + neu_pct[i] + up_pct[i] / 2, f"{up_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Label Proportions by Horizon and Threshold Rule (Train Only)", fontsize=13, y=0.99)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig



def plot_full_period_volume_trades(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot volume, number of trades, and taker buy ratio over the full period."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1. Quote Volume ($M)
    quote_vol_m = df["quote_volume"] / 1e6
    ax1.plot(df.index, quote_vol_m, color="#2962FF", linewidth=0.7, alpha=0.8)
    ax1.set_ylabel("Volume ($M / 5m)")
    ax1.set_title("Trading Activity — Full Period Overview")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Number of Trades
    trades_k = df["number_of_trades"] / 1e3
    ax2.plot(df.index, trades_k, color="#7B1FA2", linewidth=0.7, alpha=0.8)
    ax2.set_ylabel("Trades (k / 5m)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Taker Buy Ratio (Rolling 24h)
    taker_ratio = (df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)).rolling(288).mean()
    ax3.plot(df.index, taker_ratio, color="#00897B", linewidth=1.0)
    ax3.axhline(0.50, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.set_ylabel("Taker Buy Ratio\n(24h MA)")
    ax3.set_xlabel("Date (UTC)")
    ax3.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_candlestick_window(
    df_window: pd.DataFrame,
    title: str = "Candlestick Window",
    ax: Optional[plt.Axes] = None,
) -> None:
    """Render crisp Japanese candlesticks using matplotlib vector lines."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    dates = mdates.date2num(df_window.index.to_pydatetime())
    width = 0.6 / (24 * 12)  # Candle body width in days

    for i in range(len(df_window)):
        o = df_window["open"].iloc[i]
        h = df_window["high"].iloc[i]
        l = df_window["low"].iloc[i]
        c = df_window["close"].iloc[i]
        t = dates[i]

        color = "#26A69A" if c >= o else "#EF5350"  # Green / Red

        # Wick
        ax.plot([t, t], [l, h], color=color, linewidth=1.0)
        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        rect = plt.Rectangle((t - width / 2, body_bottom), width, body_height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_ylabel("Price (USDT)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_candlestick_samples(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 3 representative candlestick windows: normal day, Aug 2024 crash, Oct 2025 event."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Window 1: Normal Trading Day (2024-05-15)
    w1 = df.loc["2024-05-15 00:00:00+00:00":"2024-05-15 12:00:00+00:00"]
    plot_candlestick_window(w1, title="Sample Window 1: Typical Trading Range (May 15, 2024)", ax=ax1)

    # Window 2: August 2024 Global Macro / Yen Unwind Crash
    w2 = df.loc["2024-08-05 02:00:00+00:00":"2024-08-05 16:00:00+00:00"]
    plot_candlestick_window(w2, title="Sample Window 2: Flash Crash & V-Reversal (August 5, 2024)", ax=ax2)

    # Window 3: October 2025 Liquidation Cascade
    w3 = df.loc["2025-10-10 18:00:00+00:00":"2025-10-11 04:00:00+00:00"]
    plot_candlestick_window(w3, title="Sample Window 3: Extreme Volatility Cascade (October 10, 2025)", ax=ax3)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_data_quality_visuals(
    df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot data quality diagnostics: daily candle completeness and returns distribution with anomalies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # 1. Daily Candle Count
    daily_counts = df["close"].resample("1D").count()
    ax1.plot(daily_counts.index, daily_counts.values, color="#2E7D32", linewidth=1.0)
    ax1.axhline(288, color="#D32F2F", linestyle="--", linewidth=0.9, label="Expected: 288 candles/day")
    ax1.set_ylabel("Candles per Day")
    ax1.set_xlabel("Date")
    ax1.set_title("Data Completeness Grid (Flat at 288 = Zero Missing Days)")
    ax1.set_ylim(280, 295)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Return Histogram with Flagged Extreme Anomalies
    ret = np.log(df["close"] / df["close"].shift(1)).dropna() * 100.0
    ax2.hist(ret, bins=150, color="#1976D2", alpha=0.7, label="Standard 5m Returns")
    if not anomalies_df.empty:
        anom_ret = anomalies_df["log_return_pct"]
        ax2.scatter(anom_ret, np.ones(len(anom_ret)) * 50, color="#D32F2F", s=15, alpha=0.6, label=f"Flagged Anomalies ({len(anom_ret):,})")

    ax2.set_yscale("log")
    ax2.set_xlabel("5m Return (%)")
    ax2.set_ylabel("Frequency (Log Scale)")
    ax2.set_title("Return Distribution & Flagged Anomalies (Log Density)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# =====================================================================
# NOTEBOOK 01 PLOTS (EDA — STRICTLY ON TRAIN SPLIT)
# =====================================================================

def plot_train_price_with_ma(
    df_train: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training set closing price with 50 and 200 candle moving averages (linear & log)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    c = df_train["close"]
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()

    # Linear
    ax1.plot(c.index, c, color="#1E88E5", linewidth=1.0, label="Close Price")
    ax1.plot(ma50.index, ma50, color="#FFB300", linewidth=1.1, label="50-Candle SMA (4.2h)")
    ax1.plot(ma200.index, ma200, color="#E53935", linewidth=1.2, label="200-Candle SMA (16.7h)")
    ax1.set_ylabel("Price (USDT)")
    ax1.set_title("Training Set Price Action & Moving Averages (Linear Scale)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Log
    ax2.plot(c.index, c, color="#1E88E5", linewidth=1.0, label="Close Price (Log)")
    ax2.plot(ma50.index, ma50, color="#FFB300", linewidth=1.1, label="50-Candle SMA")
    ax2.plot(ma200.index, ma200, color="#E53935", linewidth=1.2, label="200-Candle SMA")
    ax2.set_yscale("log")
    ax2.set_ylabel("Price (Log Scale)")
    ax2.set_xlabel("Date (UTC)")
    ax2.set_title("Training Set Price Action (Log Scale)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_return_distributions(
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot return distribution: histogram with Gaussian fit, QQ-plot, and log-density."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    r_pct = returns_series.dropna() * 100.0
    mu, sigma = r_pct.mean(), r_pct.std()

    # 1. Histogram + Normal Curve
    count, bins, _ = ax1.hist(r_pct, bins=100, density=True, color="#1976D2", alpha=0.7, label="Empirical Returns")
    x = np.linspace(bins[0], bins[-1], 200)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), color="#D32F2F", linewidth=1.5, label="Normal Fit")
    ax1.set_xlim(-4, 4)
    ax1.set_xlabel("5m Return (%)")
    ax1.set_ylabel("Density")
    ax1.set_title("Return Histogram vs Normal Fit")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. QQ-Plot
    stats.probplot(r_pct, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#1976D2")
    ax2.get_lines()[0].set_markersize(2.5)
    ax2.get_lines()[1].set_color("#D32F2F")
    ax2.set_title("Normal Q-Q Plot (Pronounced Fat Tails)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Log Density
    ax3.hist(r_pct, bins=120, color="#7B1FA2", alpha=0.75, log=True)
    ax3.set_xlabel("5m Return (%)")
    ax3.set_ylabel("Count (Log Scale)")
    ax3.set_title("Log-Density Histogram (Power-Law Tails)")
    ax3.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_tail_zoom(
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot zoomed-in histograms of the extreme negative and positive return tails."""
    s = returns_series.dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    q01 = float(s.quantile(0.01))
    q99 = float(s.quantile(0.99))
    
    left_tail = s[s <= q01]
    right_tail = s[s >= q99]
    
    ax1.hist(left_tail * 100.0, bins=40, color="#C62828", alpha=0.75, edgecolor="none")
    ax1.set_title(f"Left Tail Zoom (Lowest 1%, returns <= {q01*100:.2f}%)")
    ax1.set_xlabel("Return (%)")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=":", alpha=0.4)
    
    ax2.hist(right_tail * 100.0, bins=40, color="#2E7D32", alpha=0.75, edgecolor="none")
    ax2.set_title(f"Right Tail Zoom (Highest 1%, returns >= {q99*100:.2f}%)")
    ax2.set_xlabel("Return (%)")
    ax2.set_ylabel("Count")
    ax2.grid(True, linestyle=":", alpha=0.4)
    
    fig.suptitle("Zoom on Extreme 1% Tails of Train 5m Returns (N=220,953)", fontsize=12, y=0.99)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_volatility_and_acf(
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot rolling 24h volatility and autocorrelation functions for returns, squared returns, and absolute returns."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))

    # 1. Rolling 24h Volatility
    roll_vol = (returns_series.rolling(288).std() * np.sqrt(288 * 365)) * 100.0
    ax1.plot(roll_vol.index, roll_vol, color="#E65100", linewidth=1.0)
    ax1.set_ylabel("Annualized Volatility (%)")
    ax1.set_xlabel("Date")
    ax1.set_title("24h Rolling Annualized Volatility (Train)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Autocorrelation Lags (1 to 24)
    lags = range(1, 25)
    ret_acf = [returns_series.autocorr(lag=i) for i in lags]
    sq_acf = [(returns_series ** 2).autocorr(lag=i) for i in lags]
    abs_acf = [returns_series.abs().autocorr(lag=i) for i in lags]

    # 2. Return ACF
    ax2.bar(lags, ret_acf, color="#2E7D32", width=0.6)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ci = 1.96 / np.sqrt(len(returns_series))
    ax2.axhline(ci, color="gray", linestyle="--", linewidth=0.8)
    ax2.axhline(-ci, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Lag (5m candles)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Return Autocorrelation (Near Zero)")
    ax2.set_ylim(-0.06, 0.06)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Squared & Absolute Return ACF (ARCH Effect)
    ax3.plot(lags, sq_acf, marker="o", color="#C2185B", linewidth=1.2, label="Squared Returns $r^2$")
    ax3.plot(lags, abs_acf, marker="s", color="#7B1FA2", linewidth=1.2, label="Absolute Returns $|r|$")
    ax3.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax3.set_xlabel("Lag (5m candles)")
    ax3.set_ylabel("Autocorrelation")
    ax3.set_title("Volatility Clustering / ARCH Effect")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_time_patterns(
    df_train: pd.DataFrame,
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot diurnal seasonality: volume, volatility, and return by hour, weekday, and hour x weekday heatmap."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    df_t = df_train.copy()
    df_t["hour"] = df_t.index.hour
    df_t["weekday"] = df_t.index.weekday
    df_t["abs_ret"] = returns_series.abs() * 100.0

    # 1. By UTC Hour
    h_vol = df_t.groupby("hour")["abs_ret"].mean()
    h_vol_usd = df_t.groupby("hour")["quote_volume"].mean() / 1e6
    ax1.bar(h_vol.index - 0.2, h_vol, width=0.4, color="#1976D2", label="Mean |Return| (%)")
    ax1_twin = ax1.twinx()
    ax1_twin.bar(h_vol_usd.index + 0.2, h_vol_usd, width=0.4, color="#FF9800", alpha=0.8, label="Quote Vol ($M)")
    ax1.set_xlabel("UTC Hour")
    ax1.set_ylabel("Mean |Return| (%)", color="#1976D2")
    ax1_twin.set_ylabel("Quote Vol ($M)", color="#FF9800")
    ax1.set_title("Intraday Seasonality by Hour")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. By Weekday
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    w_vol = df_t.groupby("weekday")["abs_ret"].mean()
    ax2.bar(days, w_vol.values, color="#388E3C", width=0.5)
    ax2.set_xlabel("Weekday")
    ax2.set_ylabel("Mean |Return| (%)")
    ax2.set_title("Weekly Volatility Pattern (Weekend Dip)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Heatmap Hour x Weekday
    pivot = df_t.pivot_table(index="hour", columns="weekday", values="abs_ret", aggfunc="mean")
    im = ax3.imshow(pivot, aspect="auto", cmap="YlOrRd", origin="lower")
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(days)
    ax3.set_ylabel("UTC Hour")
    ax3.set_xlabel("Weekday")
    ax3.set_title("Volatility Heatmap (Hour x Weekday)")
    plt.colorbar(im, ax=ax3, label="Mean |Return| (%)")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_microstructure_relationships(
    df_train: pd.DataFrame,
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot microstructure relationships: Volume vs |Return| and Taker Buy Ratio vs Next Return."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # 1. Volume Deciles vs Mean |Return|
    vol_bins = pd.qcut(df_train["volume"], q=10, labels=False)
    abs_ret = (returns_series.abs() * 100.0)
    vol_impact = abs_ret.groupby(vol_bins).mean()
    ax1.plot(range(1, 11), vol_impact.values, marker="o", color="#1565C0", linewidth=1.5)
    ax1.set_xlabel("Volume Decile (1=Lowest, 10=Highest)")
    ax1.set_ylabel("Mean |Return| (%)")
    ax1.set_title("Price Impact vs Volume Decile")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Taker Buy Ratio Decile vs Next Return
    taker_ratio = (df_train["taker_buy_base_volume"] / df_train["volume"].replace(0, np.nan)).dropna()
    next_ret = returns_series.shift(-1) * 100.0
    common_idx = taker_ratio.index.intersection(next_ret.dropna().index)
    tb_bins = pd.qcut(taker_ratio.loc[common_idx], q=10, labels=False)
    tb_next_ret = next_ret.loc[common_idx].groupby(tb_bins).mean()
    ax2.bar(range(1, 11), tb_next_ret.values, color=np.where(tb_next_ret.values >= 0, "#2E7D32", "#C62828"), width=0.6)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Taker Buy Ratio Decile (1=Net Sell, 10=Net Buy)")
    ax2.set_ylabel("Next Candle Mean Return (%)")
    ax2.set_title("Order Flow Predictive Lead (Taker Decile vs $R_{t+1}$)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_correlation_heatmap(
    df_train: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot correlation heatmap of base market columns and returns on Train set only."""
    fig, ax = plt.subplots(figsize=(9, 7))

    cols = ["open", "high", "low", "close", "volume", "quote_volume", "number_of_trades", "taker_buy_base_volume"]
    df_sub = df_train[cols].copy()
    df_sub["return_5m"] = np.log(df_sub["close"] / df_sub["close"].shift(1))
    df_sub["spread_hl"] = (df_sub["high"] - df_sub["low"]) / df_sub["close"]

    corr = df_sub.corr()

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap — Base Variables (Train Only)")

    # Text annotations
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_regimes_monthly(
    df: pd.DataFrame,
    returns_series: Optional[pd.Series] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot monthly return and volatility across the training period to visualize regime evolution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    if "return_pct" in df.columns and "volatility_pct" in df.columns:
        # Precomputed monthly DataFrame from eda_pipeline
        x_dates = pd.to_datetime(df["month"].astype(str)) if "month" in df.columns else df.index
        monthly_ret = df["return_pct"]
        monthly_vol = df["volatility_pct"]
    else:
        # Raw df_train
        monthly_ret = df["close"].resample("1ME").last().pct_change() * 100.0
        if returns_series is None:
            returns_series = np.log(df["close"] / df["close"].shift(1))
        monthly_vol = (returns_series.resample("1ME").std() * np.sqrt(288 * 365)) * 100.0
        x_dates = monthly_ret.index

    # Monthly return bar chart
    colors = np.where(monthly_ret >= 0, "#2E7D32", "#C62828")
    ax1.bar(x_dates, monthly_ret.values, width=20, color=colors, alpha=0.85)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("Monthly Return (%)")
    ax1.set_title("Monthly Return Regimes Across Training Set (2023 - 2025)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Monthly annualized volatility
    ax2.plot(x_dates, monthly_vol.values, marker="o", color="#E65100", linewidth=1.5)
    ax2.set_ylabel("Ann. Volatility (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Monthly Volatility Regime Dynamics")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_forward_return_horizons(
    df_train: pd.DataFrame,
    horizons: List[int] = [2, 3, 6, 12],
    dead_zone: float = config.LABEL_DEAD_ZONE,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot forward-return distributions for candidate horizons with the dead zone marked."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    close = df_train["close"].to_numpy()
    dz_pct = dead_zone * 100.0

    for i, h in enumerate(horizons):
        ax = axes[i]
        fwd_ret = ((close[h:] / close[:-h]) - 1.0) * 100.0
        inside_share = (np.abs(fwd_ret) <= dz_pct).mean() * 100.0

        ax.hist(fwd_ret, bins=120, color="#1976D2", alpha=0.75, density=True)
        ax.axvline(dz_pct, color="#D32F2F", linestyle="--", linewidth=1.3, label=f"+{dz_pct:.2f}% Dead Zone")
        ax.axvline(-dz_pct, color="#D32F2F", linestyle="--", linewidth=1.3, label=f"-{dz_pct:.2f}% Dead Zone")
        ax.axvspan(-dz_pct, dz_pct, color="#D32F2F", alpha=0.10)

        ax.set_xlim(-2.5, 2.5)
        ax.set_xlabel("Forward Return (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"Horizon = {h} Candles ({h * 5} min) — Neutral Share: {inside_share:.1f}%")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ecdf_forward_returns(
    df_train: pd.DataFrame,
    horizons: List[int] = [2, 3, 6, 12],
    cost: float = config.TOTAL_ROUNDTRIP_COST,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Empirical Cumulative Distribution Function (ECDF) of |fwd_ret_N| per horizon with line at c."""
    fig, ax = plt.subplots(figsize=(10, 6))
    close = df_train["close"].to_numpy()
    colors = ["#1E88E5", "#00897B", "#FB8C00", "#D81B60"]
    cost_pct = cost * 100.0

    for idx, N in enumerate(horizons):
        fwd_ret = (close[N:] / close[:-N]) - 1.0
        abs_ret_pct = np.sort(np.abs(fwd_ret) * 100.0)
        n = len(abs_ret_pct)
        y = np.arange(1, n + 1) / n * 100.0

        # Share below cost
        below_cost_share = float((abs_ret_pct <= cost_pct).mean() * 100.0)
        ax.plot(abs_ret_pct, y, label=f"N={N} ({N*5}m): {below_cost_share:.1f}% below cost", color=colors[idx], linewidth=1.8)

    ax.axvline(cost_pct, color="black", linestyle="--", linewidth=1.5, label=f"Cost Floor c = {cost_pct:.2f}%")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Absolute Forward Return |fwd_ret_N| (%)")
    ax.set_ylabel("Empirical Cumulative Probability (%)")
    ax.set_title("ECDF of Absolute Forward Returns Across Horizons (Train Only, N=220,953)", fontsize=12, pad=10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_label_balance_bars(
    df_train: pd.DataFrame,
    horizons: List[int] = [2, 3, 6, 12],
    dead_zone: float = config.LABEL_DEAD_ZONE,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot bar chart comparing Up, Down, and Neutral proportions across candidate horizons."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    close = df_train["close"].to_numpy()
    dz_pct = dead_zone

    up_shares, down_shares, neutral_shares = [], [], []
    labels = []

    for h in horizons:
        fwd = (close[h:] / close[:-h]) - 1.0
        n_valid = len(fwd)
        u = (fwd > dz_pct).sum() / n_valid * 100.0
        d = (fwd < -dz_pct).sum() / n_valid * 100.0
        neu = (np.abs(fwd) <= dz_pct).sum() / n_valid * 100.0

        up_shares.append(u)
        down_shares.append(d)
        neutral_shares.append(neu)
        labels.append(f"{h}c ({h*5}m)")

    x = np.arange(len(horizons))
    width = 0.25

    ax.bar(x - width, up_shares, width, label="Up (+1)", color="#2E7D32")
    ax.bar(x, neutral_shares, width, label="Neutral (0, Dead Zone)", color="#757575")
    ax.bar(x + width, down_shares, width, label="Down (-1)", color="#C62828")

    for i in range(len(horizons)):
        ax.text(x[i] - width, up_shares[i] + 1, f"{up_shares[i]:.1f}%", ha="center", fontsize=8)
        ax.text(x[i], neutral_shares[i] + 1, f"{neutral_shares[i]:.1f}%", ha="center", fontsize=8)
        ax.text(x[i] + width, down_shares[i] + 1, f"{down_shares[i]:.1f}%", ha="center", fontsize=8)

    ax.set_ylabel("Share of Candles (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title(f"Label Class Balance Across Candidate Horizons (Dead Zone = {dead_zone * 100:.2f}%)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# =====================================================================
# 40-BAR LABEL DISTRIBUTION PLOTS & SECTION B BIG_MOVE PLOTS
# =====================================================================

def plot_label_counts_all(
    balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ONE stacked bar per label column (40 bars across 10 rules): Up green, Down red, Neutral grey.
    
    Annotates integer counts directly on the bar segments.
    """
    fig, ax = plt.subplots(figsize=(22, 7.5))
    labels = balance_df["label_col"].tolist()
    up_cnt = balance_df["n_up"].to_numpy()
    dn_cnt = balance_df["n_down"].to_numpy()
    neu_cnt = balance_df["n_neutral"].to_numpy()
    
    x = np.arange(len(labels))
    w = 0.75

    p_dn = ax.bar(x, dn_cnt, w, label="Down (-1)", color="#C62828", alpha=0.9)
    p_neu = ax.bar(x, neu_cnt, w, bottom=dn_cnt, label="Neutral (0)", color="#9E9E9E", alpha=0.8)
    p_up = ax.bar(x, up_cnt, w, bottom=dn_cnt + neu_cnt, label="Up (+1)", color="#2E7D32", alpha=0.9)

    ax.set_xticks(x)
    formatted_labels = []
    for l in labels:
        clean_l = l.replace("target_", "")
        if any(sr in clean_l for sr in ["F05", "F10", "F15"]):
            formatted_labels.append(f"{clean_l}*")
        else:
            formatted_labels.append(clean_l)
    ax.set_xticklabels(formatted_labels, rotation=70, ha="right", fontsize=8.5)
    ax.set_ylabel("Candle Count")
    ax.set_title("Label Class Counts Across All 40 Target Columns (* = Sub-Cost Rule, Train Only, N=220,953)", fontsize=13, pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for i in range(len(labels)):
        if dn_cnt[i] > 10000:
            ax.text(i, dn_cnt[i] / 2, f"{dn_cnt[i]:,}", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        if neu_cnt[i] > 20000:
            ax.text(i, dn_cnt[i] + neu_cnt[i] / 2, f"{neu_cnt[i]:,}", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        if up_cnt[i] > 10000:
            ax.text(i, dn_cnt[i] + neu_cnt[i] + up_cnt[i] / 2, f"{up_cnt[i]:,}", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_label_percent_all(
    balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ONE stacked bar per label column (40 bars) normalized to 100%.
    
    Annotates percentages directly on the segments.
    """
    fig, ax = plt.subplots(figsize=(22, 7.5))
    labels = balance_df["label_col"].tolist()
    up_pct = balance_df["pct_up"].to_numpy()
    dn_pct = balance_df["pct_down"].to_numpy()
    neu_pct = balance_df["pct_neutral"].to_numpy()
    
    x = np.arange(len(labels))
    w = 0.75

    p_dn = ax.bar(x, dn_pct, w, label="Down (-1)", color="#C62828", alpha=0.9)
    p_neu = ax.bar(x, neu_pct, w, bottom=dn_pct, label="Neutral (0)", color="#9E9E9E", alpha=0.8)
    p_up = ax.bar(x, up_pct, w, bottom=dn_pct + neu_pct, label="Up (+1)", color="#2E7D32", alpha=0.9)

    ax.set_xticks(x)
    formatted_labels = []
    for l in labels:
        clean_l = l.replace("target_", "")
        if any(sr in clean_l for sr in ["F05", "F10", "F15"]):
            formatted_labels.append(f"{clean_l}*")
        else:
            formatted_labels.append(clean_l)
    ax.set_xticklabels(formatted_labels, rotation=70, ha="right", fontsize=8.5)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Label Class Proportions (100% Stacked) Across All 40 Rules (* = Sub-Cost Rule, Train Only)", fontsize=13, pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for i in range(len(labels)):
        if dn_pct[i] > 5.0:
            ax.text(i, dn_pct[i] / 2, f"{dn_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        if neu_pct[i] > 10.0:
            ax.text(i, dn_pct[i] + neu_pct[i] / 2, f"{neu_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        if up_pct[i] > 5.0:
            ax.text(i, dn_pct[i] + neu_pct[i] + up_pct[i] / 2, f"{up_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_label_by_horizon(
    balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 4 panels (one per horizon N in {2,3,6,12}), each showing 10 rules with sub-cost hatching."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    horizons = [2, 3, 6, 12]
    
    for idx, N in enumerate(horizons):
        ax = axes[idx]
        sub = balance_df[balance_df["horizon"] == N].copy()
        rules = sub["theta_rule"].tolist()
        up_pct = sub["pct_up"].to_numpy()
        dn_pct = sub["pct_down"].to_numpy()
        neu_pct = sub["pct_neutral"].to_numpy()
        
        x = np.arange(len(rules))
        w = 0.65

        for i, r in enumerate(rules):
            is_sub = (r in ["F05", "F10", "F15"])
            hatch = "//" if is_sub else None
            edgecolor = "black" if is_sub else "none"

            ax.bar(x[i], dn_pct[i], w, color="#C62828", alpha=0.9, hatch=hatch, edgecolor=edgecolor)
            ax.bar(x[i], neu_pct[i], w, bottom=dn_pct[i], color="#9E9E9E", alpha=0.8, hatch=hatch, edgecolor=edgecolor)
            ax.bar(x[i], up_pct[i], w, bottom=dn_pct[i] + neu_pct[i], color="#2E7D32", alpha=0.9, hatch=hatch, edgecolor=edgecolor)

        formatted_rules = [f"{r}*" if r in ["F05", "F10", "F15"] else r for r in rules]
        ax.set_xticks(x)
        ax.set_xticklabels(formatted_rules, fontsize=8.5)
        ax.set_title(f"Horizon N={N} ({N*5}m) — Hatched (*): Sub-Cost", fontsize=11, fontweight="bold")
        ax.set_ylabel("Share (%)" if idx % 2 == 0 else "")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for i in range(len(rules)):
            if dn_pct[i] > 5.0:
                ax.text(i, dn_pct[i] / 2, f"{dn_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
            if neu_pct[i] > 10.0:
                ax.text(i, dn_pct[i] + neu_pct[i] / 2, f"{neu_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
            if up_pct[i] > 5.0:
                ax.text(i, dn_pct[i] + neu_pct[i] + up_pct[i] / 2, f"{up_pct[i]:.1f}%", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", label="Up (+1)"),
        Patch(facecolor="#9E9E9E", label="Neutral (0)"),
        Patch(facecolor="#C62828", label="Down (-1)"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Sub-Cost Rule (*)"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig.suptitle("Label Proportions by Horizon and Rule with Sub-Cost Rules Highlighted (Train Only)", fontsize=13, y=0.99)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Section B: Big Move Target Visualization Functions
# ---------------------------------------------------------------------------

def plot_big_move_balance(
    bm_balance_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot stacked bar chart of big_move_1 vs big_move_0 shares across 4 horizons."""
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = bm_balance_df["horizon"].tolist()
    pct_ones = bm_balance_df["pct_big_move_1"].to_numpy()
    pct_zeros = bm_balance_df["pct_big_move_0"].to_numpy()
    ones_cnt = bm_balance_df["big_move_1"].to_numpy()
    zeros_cnt = bm_balance_df["big_move_0"].to_numpy()

    x = np.arange(len(horizons))
    w = 0.55

    ax.bar(x, pct_ones, w, label="Big Move = 1 (|fwd_ret| > c)", color="#5E35B1", alpha=0.9)
    ax.bar(x, pct_zeros, w, bottom=pct_ones, label="Big Move = 0 (|fwd_ret| <= c)", color="#CFD8DC", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={N} ({N*5}m)" for N in horizons], fontsize=10, fontweight="bold")
    ax.set_ylabel("Share of Valid Candles (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Big Move Target Class Balance Across Horizons (Train Only, N=220,953)", fontsize=12, pad=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for i in range(len(horizons)):
        ax.text(i, pct_ones[i] / 2, f"{pct_ones[i]:.1f}%\n({ones_cnt[i]:,})", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(i, pct_ones[i] + pct_zeros[i] / 2, f"{pct_zeros[i]:.1f}%\n({zeros_cnt[i]:,})", ha="center", va="center", color="#37474F", fontsize=9, fontweight="bold")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_big_move_monthly(
    monthly_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot monthly incidence of big_move_N alongside average monthly rolling volatility."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    months = monthly_df["month"].tolist()
    x = np.arange(len(months))

    colors = {"2": "#1E88E5", "3": "#00897B", "6": "#FB8C00", "12": "#7B1FA2"}
    for N in [2, 3, 6, 12]:
        col = f"big_move_{N}_pct"
        if col in monthly_df.columns:
            ax1.plot(x, monthly_df[col], marker="o", label=f"N={N} ({N*5}m)", color=colors[str(N)], linewidth=1.8)

    ax1.set_ylabel("Big Move Rate (%)", fontsize=10)
    ax1.set_title("Monthly Big Move Incidence Across Candidate Horizons (Train)", fontsize=12)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    if "avg_sigma_288" in monthly_df.columns:
        ax2.bar(x, monthly_df["avg_sigma_288"], color="#E65100", alpha=0.75, width=0.6, label="Avg Rolling Vol (feat_vol_sigma_288 %)")
        ax2.set_ylabel("Rolling Vol (%)", fontsize=10)
        ax2.set_title("Monthly Mean 24-Hour Realized Volatility", fontsize=11)
        ax2.legend(loc="upper left", framealpha=0.9)
        ax2.grid(True, linestyle=":", alpha=0.5)

    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=9)
    ax2.set_xlabel("Month (UTC)", fontsize=10)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_big_move_context(
    context_dict: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot big move incidence across intraday hour, day of week, and volume terciles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df_hour = context_dict["by_hour"]
    df_dow = context_dict["by_dow"]
    df_terc = context_dict["by_tercile"]

    colors = {"2": "#1E88E5", "3": "#00897B", "6": "#FB8C00", "12": "#7B1FA2"}

    # 1. Hour of day
    ax1 = axes[0]
    for N in [2, 3, 6, 12]:
        ax1.plot(df_hour["hour"], df_hour[f"big_move_{N}_pct"], marker=".", label=f"N={N}", color=colors[str(N)], linewidth=1.5)
    ax1.set_title("By Hour of Day (UTC)", fontsize=11)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Big Move Rate (%)")
    ax1.set_xticks(range(0, 24, 3))
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8)

    # 2. Day of week
    ax2 = axes[1]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for N in [2, 3, 6, 12]:
        ax2.plot(range(7), df_dow[f"big_move_{N}_pct"], marker="o", label=f"N={N}", color=colors[str(N)], linewidth=1.5)
    ax2.set_title("By Day of Week", fontsize=11)
    ax2.set_xlabel("Weekday")
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(days)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8)

    # 3. Volume tercile
    ax3 = axes[2]
    x_terc = np.arange(len(df_terc))
    w = 0.18
    for i, N in enumerate([2, 3, 6, 12]):
        ax3.bar(x_terc + (i - 1.5) * w, df_terc[f"big_move_{N}_pct"], width=w, label=f"N={N}", color=colors[str(N)], alpha=0.85)
    ax3.set_title("By Traded Volume Tercile", fontsize=11)
    ax3.set_xticks(x_terc)
    ax3.set_xticklabels(df_terc["volume_tercile"])
    ax3.set_xlabel("Volume Tercile")
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    ax3.legend(fontsize=8)

    fig.suptitle("Big Move Incidence Conditioning by Temporal & Volume Context (Train)", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_big_move_acf(
    persistence_dict: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ACF and run-length statistics for big_move_N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    acf_data = persistence_dict["acf"]
    df_runs = persistence_dict["runs"]

    colors = {"N_2": "#1E88E5", "N_3": "#00897B", "N_6": "#FB8C00", "N_12": "#7B1FA2"}

    # Left: ACF
    for k, acf_vals in acf_data.items():
        lags = np.arange(1, len(acf_vals) + 1)
        ax1.plot(lags, acf_vals, label=k.replace("_", "="), color=colors[k], linewidth=1.6)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Autocorrelation Function of big_move_N (Lags 1..50)", fontsize=11)
    ax1.set_xlabel("Lag (5m candles)")
    ax1.set_ylabel("Autocorrelation")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: Run lengths
    horizons = df_runs["horizon"].tolist()
    x = np.arange(len(horizons))
    w = 0.35
    ax2.bar(x - w/2, df_runs["mean_run_1"], width=w, label="Mean Run (Big Move = 1)", color="#5E35B1", alpha=0.85)
    ax2.bar(x + w/2, df_runs["mean_run_0"], width=w, label="Mean Run (Quiet = 0)", color="#78909C", alpha=0.85)

    for i in range(len(horizons)):
        ax2.text(i - w/2, df_runs["mean_run_1"].iloc[i] + 0.1, f"{df_runs['mean_run_1'].iloc[i]:.1f}", ha="center", fontsize=8.5)
        ax2.text(i + w/2, df_runs["mean_run_0"].iloc[i] + 0.1, f"{df_runs['mean_run_0'].iloc[i]:.1f}", ha="center", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={N}" for N in horizons])
    ax2.set_title("Mean Consecutive Run Lengths (Candles)", fontsize=11)
    ax2.set_ylabel("Consecutive Candles")
    ax2.legend()
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_big_move_cohens_d(
    cohens_d_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of top features separating big_move_1 vs big_move_0 by Cohen's d."""
    top_df = cohens_d_df.head(top_n).sort_values(by="abs_cohens_d", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    group_colors = {
        "volatility": "#E65100",
        "momentum": "#1E88E5",
        "volume": "#2E7D32",
        "order_flow": "#7B1FA2",
        "candle": "#F57C00",
        "multi_timeframe": "#00838F",
        "time": "#546E7A",
    }
    bar_colors = [group_colors.get(g, "#424242") for g in top_df["group"]]

    y = np.arange(len(top_df))
    ax.barh(y, top_df["cohens_d"], color=bar_colors, alpha=0.85, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(top_df["feature"], fontsize=9)
    ax.set_xlabel("Cohen's d (Big Move 1 vs Big Move 0)", fontsize=10)
    ax.set_title(f"Top {top_n} Engineered Features by Separation Power for big_move_12", fontsize=12, pad=12)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Legend for groups
    from matplotlib.patches import Patch
    used_groups = top_df["group"].unique()
    legend_elements = [Patch(facecolor=group_colors.get(g, "#424242"), label=g.title()) for g in used_groups]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_big_move_vol_deciles(
    decile_df: pd.DataFrame,
    auc_scores: Dict[str, float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot big move incidence across volatility deciles with Stage 1 AUC benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(decile_df))
    w = 0.18
    colors = {"2": "#1E88E5", "3": "#00897B", "6": "#FB8C00", "12": "#7B1FA2"}

    for i, N in enumerate([2, 3, 6, 12]):
        col = f"big_move_{N}_pct"
        auc = auc_scores.get(f"N_{N}", 0.0)
        ax.bar(x + (i - 1.5) * w, decile_df[col], width=w, label=f"N={N} (AUC = {auc:.3f})", color=colors[str(N)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}\n({decile_df['vol_mean'].iloc[d-1]:.2f}%)" for d in range(1, 11)], fontsize=8.5)
    ax.set_xlabel("Volatility Decile (Mean Rolling 24h Sigma %)", fontsize=10)
    ax.set_ylabel("Big Move Rate (%)", fontsize=10)
    ax.set_title("Big Move Incidence Monotonically Rising Across Volatility Deciles (Stage 1 Predictability)", fontsize=12, pad=12)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
    return fig


def plot_feature_group_histograms(
    df_features: pd.DataFrame,
    group_name: str,
    feature_cols: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot grid of histograms for features in a specific feature group."""
    n_cols = len(feature_cols)
    nrows = int(np.ceil(n_cols / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        vals = df_features[col].dropna()
        q01, q99 = vals.quantile(0.005), vals.quantile(0.995)
        clipped = vals.clip(q01, q99)
        ax.hist(clipped, bins=45, color="#1976D2", alpha=0.75, edgecolor="none")
        ax.set_title(col, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distributions of Features in Group: {group_name.upper()} (Train Only)", fontsize=12, y=0.99)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rolling_macro_trends(
    df_train: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot rolling 24h volatility, rolling 30-day cumulative return, and rolling 30-day mean volume."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    c = df_train["close"]
    v = df_train["volume"]
    ret_5m = np.log(c / c.shift(1))

    # 1. Rolling 24h volatility (annualized)
    roll_vol = (ret_5m.rolling(288).std() * np.sqrt(288 * 365)) * 100.0
    ax1.plot(roll_vol.index, roll_vol, color="#E65100", linewidth=1.1)
    ax1.set_ylabel("Ann. Volatility (%)")
    ax1.set_title("24-Hour Rolling Annualized Volatility (Train)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Rolling 30-day return
    w_30d = 288 * 30
    roll_30d_ret = (c / c.shift(w_30d) - 1.0) * 100.0
    ax2.plot(roll_30d_ret.index, roll_30d_ret, color="#1E88E5", linewidth=1.1)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("30-Day Return (%)")
    ax2.set_title("Rolling 30-Day Total Return")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Rolling 30-day mean volume
    roll_vol_mean = v.rolling(w_30d).mean()
    ax3.plot(roll_vol_mean.index, roll_vol_mean, color="#388E3C", linewidth=1.1)
    ax3.set_ylabel("Mean Volume (BTC)")
    ax3.set_xlabel("Date (UTC)")
    ax3.set_title("Rolling 30-Day Mean 5m Traded Volume")
    ax3.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_acf_pacf_multi(
    df_train: pd.DataFrame,
    nlags: int = 288,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot sample ACF up to lag 288 for returns, abs returns, squared returns, and volume."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    ret_5m = np.log(df_train["close"] / df_train["close"].shift(1)).dropna()
    abs_ret = ret_5m.abs()
    sq_ret = ret_5m ** 2
    vol = df_train["volume"].dropna()

    series_list = [
        ("Log Returns (Near Zero Autocorrelation)", ret_5m, axes[0, 0], "#1E88E5"),
        ("Absolute Returns (Persistent Volatility Clustering)", abs_ret, axes[0, 1], "#FB8C00"),
        ("Squared Returns (ARCH Effect)", sq_ret, axes[1, 0], "#D81B60"),
        ("Volume (Diurnal 288-Lag Autocorrelation)", vol, axes[1, 1], "#00897B"),
    ]

    lags = np.arange(1, min(nlags + 1, 289))
    ci = 1.96 / np.sqrt(len(ret_5m))

    for title, s, ax, color in series_list:
        from src.eda import compute_acf_manual
        acf_vals = compute_acf_manual(s, nlags=len(lags))[1:]
        ax.plot(lags, acf_vals, color=color, linewidth=1.2)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.axhline(ci, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(-ci, color="gray", linestyle="--", linewidth=0.8)
        # Highlight 288 lag (24h)
        if len(lags) >= 288:
            ax.axvline(288, color="red", linestyle=":", alpha=0.6, label="Lag 288 (24h)")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lag (5m candles)")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_seasonality_heatmaps(
    df_train: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 2D heatmap of hour-of-day x weekday for return, abs return, volume, trades."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ret_5m = np.log(df_train["close"] / df_train["close"].shift(1)) * 100.0
    abs_ret = ret_5m.abs()
    vol = df_train["volume"]
    trades = df_train["number_of_trades"]

    hour = df_train.index.hour
    dow = df_train.index.dayofweek if hasattr(df_train.index, "dayofweek") else df_train.index.day_of_week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    metrics = [
        ("Mean Return (%)", ret_5m, axes[0, 0], "coolwarm"),
        ("Mean Absolute Return (%)", abs_ret, axes[0, 1], "YlOrRd"),
        ("Mean Volume (BTC)", vol, axes[1, 0], "viridis"),
        ("Mean Trades Count", trades, axes[1, 1], "magma"),
    ]

    for title, s, ax, cmap in metrics:
        t_df = pd.DataFrame({"h": hour, "d": dow, "val": s})
        pivot = t_df.pivot_table(index="d", columns="h", values="val", aggfunc="mean")
        im = ax.imshow(pivot.to_numpy(), cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_xlabel("Hour (UTC)")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_threshold_sweep(
    df_train: pd.DataFrame,
    horizons: List[int] = [2, 3, 6, 12],
    cost: float = config.TOTAL_ROUNDTRIP_COST,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Up / Down / Neutral shares across a range of theta thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    thetas = np.arange(0.0005, 0.0065, 0.0005)
    close = df_train["close"].to_numpy()

    for idx, N in enumerate(horizons):
        ax = axes[idx]
        fwd_ret = (close[N:] / close[:-N]) - 1.0
        n_valid = len(fwd_ret)

        up_shares, down_shares, neut_shares = [], [], []
        for th in thetas:
            u = (fwd_ret > th).sum() / n_valid * 100.0
            d = (fwd_ret < -th).sum() / n_valid * 100.0
            neu = (np.abs(fwd_ret) <= th).sum() / n_valid * 100.0
            up_shares.append(u)
            down_shares.append(d)
            neut_shares.append(neu)

        th_pct = thetas * 100.0
        ax.plot(th_pct, up_shares, color="#2E7D32", label="Up Share (%)", linewidth=1.5)
        ax.plot(th_pct, down_shares, color="#C62828", label="Down Share (%)", linewidth=1.5)
        ax.plot(th_pct, neut_shares, color="#757575", label="Neutral Share (%)", linewidth=1.5)

        ax.axvline(cost * 100, color="black", linestyle="--", label=f"Cost Floor c = {cost*100:.2f}%")
        ax.axvline(0.30, color="purple", linestyle=":", label="S2 = 0.30%")

        ax.set_title(f"Horizon N={N} ({N*5}m) Sensitivity")
        ax.set_xlabel("Threshold θ (%)")
        ax.set_ylabel("Share (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rolling_feature_correlations(
    df_transforms: pd.DataFrame,
    fwd_ret_12: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 30-day rolling correlation of taker imbalance and volume with forward return."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    window = 288 * 30

    imb_col = "taker_buy_ratio" if "taker_buy_ratio" in df_transforms.columns else ("taker_imbalance" if "taker_imbalance" in df_transforms.columns else df_transforms.columns[0])
    vol_col = "log_volume" if "log_volume" in df_transforms.columns else ("feat_volume_log" if "feat_volume_log" in df_transforms.columns else df_transforms.columns[1])

    imb = df_transforms[imb_col]
    vol = df_transforms[vol_col]

    roll_corr_imb = imb.rolling(window).corr(fwd_ret_12)
    roll_corr_vol = vol.rolling(window).corr(fwd_ret_12)

    ax.plot(roll_corr_imb.index, roll_corr_imb, color="#1E88E5", label=f"Rolling Corr({imb_col}, fwd_ret_12)")
    ax.plot(roll_corr_vol.index, roll_corr_vol, color="#E53935", label=f"Rolling Corr({vol_col}, fwd_ret_12)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_title("Stability of Feature-Return Relationships (Rolling 30-Day Pearson Corr)")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date (UTC)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_correlation_annotated(
    corr: pd.DataFrame,
    title: str = "Correlation Heatmap",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot an annotated correlation heatmap from an existing correlation matrix."""
    n_rows, n_cols = corr.shape
    fig_w = max(8, min(20, n_cols * 0.7 + 3))
    fig_h = max(6, min(16, n_rows * 0.5 + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8 if n_cols > 15 else 9)
    ax.set_yticklabels(corr.index, fontsize=8 if n_rows > 15 else 9)
    ax.set_title(title, fontsize=11, pad=10)

    # Annotate if matrix is reasonably sized
    if n_rows <= 25 and n_cols <= 25:
        for i in range(n_rows):
            for j in range(n_cols):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    color = "white" if abs(val) > 0.55 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7 if max(n_rows, n_cols) > 15 else 8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# Aliases
plot_monthly_regimes = plot_regimes_monthly


# =====================================================================
# NOTEBOOK 04 PLOTS (FEATURE ENGINEERING & SELECTION EDA)
# =====================================================================

def plot_feature_correlations_clustered(
    corr: pd.DataFrame,
    title: str = "Clustered Feature Correlation Matrix (Pearson)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot hierarchical cluster-ordered correlation heatmap for all 88 features."""
    from scipy.cluster import hierarchy

    # Compute linkage on distance matrix
    dist = np.clip(1.0 - np.abs(corr.values), 0.0, 2.0)
    # Ensure symmetry
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    condensed_dist = dist[np.triu_indices(len(dist), k=1)]
    linkage = hierarchy.linkage(condensed_dist, method="average")
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro["leaves"]

    reordered_corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(reordered_corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    
    n = len(reordered_corr)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(reordered_corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(reordered_corr.index, fontsize=6)
    ax.set_title(title, fontsize=13, pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Correlation Coefficient", fontsize=10)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_size_auc_ranked_bars(
    df_auc: pd.DataFrame,
    horizons: List[int] = config.CANDIDATE_HORIZONS,
    top_k: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Ranked bar charts of single-feature ROC AUC for target big_move_N across all 4 candidate horizons."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for idx, N in enumerate(horizons):
        ax = axes[idx]
        col_name = f"auc_{N}"
        df_sorted = df_auc.sort_values(by=col_name, ascending=False).reset_index(drop=True)
        top_df = df_sorted.head(top_k).copy()

        # Check if reference feature feat_vol_sigma_288 is in top_k, if not append it
        ref_feat = "feat_vol_sigma_288"
        if ref_feat not in top_df["feature"].values and ref_feat in df_sorted["feature"].values:
            ref_row = df_sorted[df_sorted["feature"] == ref_feat]
            top_df = pd.concat([top_df, ref_row])

        # Colors: highlight reference in gold, others in blue
        colors = ["#FFA000" if f == ref_feat else "#1E88E5" for f in top_df["feature"]]
        
        y_pos = np.arange(len(top_df))
        ax.barh(y_pos, top_df[col_name], color=colors, edgecolor="black", linewidth=0.5, height=0.7)
        ax.axvline(0.50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Chance (0.50)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_df["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.45, max(0.80, top_df[col_name].max() + 0.05))
        ax.set_xlabel("ROC AUC", fontsize=9)
        ax.set_title(f"Horizon N={N} ({N*5}m) — Worth-It Target big_move_{N}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Label reference feature
        if ref_feat in top_df["feature"].values:
            ref_auc = top_df.loc[top_df["feature"] == ref_feat, col_name].values[0]
            ax.plot([], [], color="#FFA000", label=f"Ref {ref_feat} (AUC={ref_auc:.3f})")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Single-Feature Predictive Power for Size Target (big_move_N ROC AUC)", fontsize=14, y=0.99)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_size_cohens_d_heatmap(
    df_cohen: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot heatmap of Cohen's d effect sizes with target big_move_N across all 88 features."""
    cols = ["cohens_d_2", "cohens_d_3", "cohens_d_6", "cohens_d_12"]
    sub = df_cohen.set_index("feature")[cols]
    
    # Sort features by average effect size across horizons
    sub = sub.loc[sub.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, 20))
    im = ax.imshow(sub.values, cmap="magma", aspect="auto", vmin=0.0, vmax=0.75)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["N=2 (10m)", "N=3 (15m)", "N=6 (30m)", "N=12 (60m)"], fontsize=9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=6.5)
    ax.set_title("Cohen's d Effect Size: big_move=1 vs big_move=0 Across All Horizons", fontsize=11, pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's d", fontsize=9)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_direction_metrics_heatmap(
    df_dir: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot directional metrics heatmap: ROC AUC against raw_sign_N across all candidate horizons."""
    cols = ["auc_2", "auc_3", "auc_6", "auc_12"]
    sub = df_dir.set_index("feature")[cols]
    sub = sub.loc[sub.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, 20))
    im = ax.imshow(sub.values, cmap="viridis", aspect="auto", vmin=0.50, vmax=0.55)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["N=2 (10m)", "N=3 (15m)", "N=6 (30m)", "N=12 (60m)"], fontsize=9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=6.5)
    ax.set_title("Single-Feature Directional Power: ROC AUC vs raw_sign_N", fontsize=11, pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Directional ROC AUC", fontsize=9)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_lagged_flow_loc(
    df_lagged: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot lagged correlations with forward return for order flow and location features."""
    fig, ax = plt.subplots(figsize=(10, 5))
    lag_cols = [c for c in df_lagged.columns if c.startswith("lag_")]
    lags = [int(c.split("_")[1]) for c in lag_cols]

    for _, row in df_lagged.iterrows():
        feat = row["feature"]
        vals = [row[c] for c in lag_cols]
        ax.plot(lags, vals, marker="o", markersize=4, label=feat if len(df_lagged) <= 12 else None, alpha=0.7)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag k (5m candles back)", fontsize=10)
    ax.set_ylabel("Spearman Correlation with fwd_ret_12", fontsize=10)
    ax.set_title("Predictive Decay: Lagged Correlation of Flow and Location Features with fwd_ret_12", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    if len(df_lagged) <= 12:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rolling_direction_correlation(
    df_train: pd.DataFrame,
    top_features: List[str],
    fwd_ret_col: str = "fwd_ret_12",
    window_days: int = 30,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 30-day rolling correlation of top directional features with forward return."""
    fig, ax = plt.subplots(figsize=(13, 5))
    window = 288 * window_days
    fwd_ret = df_train[fwd_ret_col]

    colors = ["#1E88E5", "#E53935", "#43A047", "#8E24AA", "#FB8C00"]
    for idx, feat in enumerate(top_features[:5]):
        if feat in df_train.columns:
            roll_corr = df_train[feat].rolling(window).corr(fwd_ret)
            ax.plot(roll_corr.index, roll_corr, label=feat, color=colors[idx % len(colors)], linewidth=1.1, alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"Temporal Stability of Direction Features: Rolling {window_days}-Day Spearman Corr with {fwd_ret_col}", fontsize=11)
    ax.set_ylabel("Rolling Correlation", fontsize=10)
    ax.set_xlabel("Date (UTC)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_context_profiles(
    by_hour: pd.DataFrame,
    by_dow: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot intraday and day-of-week context profiles for key features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    vol_col = [c for c in by_hour.columns if "sigma_288" in c or "gk" in c][0]
    vol_seas_col = [c for c in by_hour.columns if "seasonal_vol" in c][0] if any("seasonal_vol" in c for c in by_hour.columns) else None

    # Hour profile
    ax1.plot(by_hour.index, by_hour[vol_col] / by_hour[vol_col].mean(), marker="o", color="#1E88E5", label="Raw Volatility (rel to mean)")
    if vol_seas_col:
        ax1.plot(by_hour.index, by_hour[vol_seas_col], marker="s", color="#43A047", label="Seasonal Vol Ratio")
    ax1.set_xlabel("UTC Hour of Day", fontsize=9)
    ax1.set_ylabel("Relative Level", fontsize=9)
    ax1.set_title("Intraday Diurnal Cycle", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Weekday profile
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax2.plot(range(len(by_dow)), by_dow[vol_col] / by_dow[vol_col].mean(), marker="o", color="#1E88E5", label="Raw Volatility")
    if vol_seas_col:
        ax2.plot(range(len(by_dow)), by_dow[vol_seas_col], marker="s", color="#43A047", label="Seasonal Vol Ratio")
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(days, fontsize=9)
    ax2.set_xlabel("Day of Week (UTC)", fontsize=9)
    ax2.set_ylabel("Relative Level", fontsize=9)
    ax2.set_title("Weekly Day-of-Week Cycle", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_acf_deseasoning(
    df_acf: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot autocorrelation function (ACF) comparing raw volume vs de-seasonalized volume ratio."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    lags = df_acf["lag"]

    ax.plot(lags, df_acf["acf_raw_volume"], color="#E53935", linewidth=1.2, label="Raw 5m Volume ACF")
    ax.plot(lags, df_acf["acf_deseasoned_volume"], color="#1E88E5", linewidth=1.2, label="7d De-Seasonalized Volume Ratio ACF")

    # Mark 24h lag (288 candles)
    ax.axvline(288, color="gray", linestyle=":", linewidth=1.0, label="24h Lag (288 candles)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Lag (5-minute candles)", fontsize=9)
    ax.set_ylabel("Autocorrelation", fontsize=9)
    ax.set_title("Elimination of Diurnal Cycle: Raw Volume vs 7-Day Same-Slot Volume Ratio", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig



def plot_direction_quarterly_stability(
    df_q_stab: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot quarterly ROC AUC heatmap for the top 15 direction features across all 9 quarters."""
    q_cols = [c for c in df_q_stab.columns if c.startswith("auc_20")]
    q_labels = [c.replace("auc_", "") for c in q_cols]
    feats = df_q_stab["feature"].tolist()
    matrix = df_q_stab[q_cols].values

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.48, vmax=0.56)

    ax.set_xticks(range(len(q_cols)))
    ax.set_xticklabels(q_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)

    # Annotate values inside cells
    for i in range(len(feats)):
        for j in range(len(q_cols)):
            val = matrix[i, j]
            text_color = "white" if val > 0.53 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title("Temporal Stability of Top 15 Direction Features: ROC AUC by Quarter (Train Partition)", fontsize=11, pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Quarterly ROC AUC", fontsize=9)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_group_correlation_heatmaps(
    df_train: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    figures_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Plot and save individual correlation heatmaps for each feature group.
    
    Saves to results/figures/04_corr_<group>.png.
    """
    if figures_dir is None:
        figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    for grp_name, cols in feature_groups.items():
        avail_cols = [c for c in cols if c in df_train.columns]
        if len(avail_cols) < 2:
            continue
        corr = df_train[avail_cols].corr(method="spearman")
        
        n_feats = len(avail_cols)
        fig_size = max(5, int(n_feats * 0.75))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(n_feats))
        ax.set_yticks(range(n_feats))
        ax.set_xticklabels(avail_cols, rotation=90, fontsize=8)
        ax.set_yticklabels(avail_cols, fontsize=8)
        ax.set_title(f"Feature Group Correlation (Spearman): {grp_name}", fontsize=10, pad=10)
        
        # Annotate if <= 10 features
        if n_feats <= 10:
            for i in range(n_feats):
                for j in range(n_feats):
                    val = corr.values[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                            color="white" if abs(val) > 0.6 else "black", fontsize=7.5)
                            
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_p = figures_dir / f"04_corr_{grp_name}.png"
        fig.savefig(out_p, dpi=150)
        plt.close(fig)
        saved_paths[grp_name] = out_p
        
    return saved_paths
