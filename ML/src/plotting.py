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
    df_train: pd.DataFrame,
    returns_series: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot monthly return and volatility across the training period to visualize regime evolution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    monthly_ret = df_train["close"].resample("1ME").last().pct_change() * 100.0
    monthly_vol = (returns_series.resample("1ME").std() * np.sqrt(288 * 365)) * 100.0

    # Monthly return bar chart
    colors = np.where(monthly_ret >= 0, "#2E7D32", "#C62828")
    ax1.bar(monthly_ret.index, monthly_ret.values, width=20, color=colors, alpha=0.85)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("Monthly Return (%)")
    ax1.set_title("Monthly Return Regimes Across Training Set (2023 - 2025)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Monthly annualized volatility
    ax2.plot(monthly_vol.index, monthly_vol.values, marker="o", color="#E65100", linewidth=1.5)
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

