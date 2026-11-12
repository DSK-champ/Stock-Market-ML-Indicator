# AI Trader Dashboard (`frontend/blank.html`)

This dashboard provides a TradingView dark-theme real-time trading terminal and held-out historical replay interface for the BTC/USDT 5-minute Machine Learning indicator ($N=2$ horizon, 10-minute forecast).

---

## 1. Quickstart & Run Commands

### Single-Command All-in-One Launcher (Recommended)
You can start all 3 services concurrently in **one single terminal** with automatic browser launch:
```bash
python start_dashboard.py
```
- Starts the Live Engine (Port 8000)
- Starts the Dashboard API (Port 8001)
- Serves the Frontend (Port 8080)
- Opens `http://localhost:8080/blank.html` in your default browser automatically.
- Press `Ctrl+C` to stop all 3 services cleanly.

**Other single-command options:**
```bash
# Accelerated historical replay at 60x speed:
python start_dashboard.py --mode replay --speed 60

# Offline demo mode with synthetic data (no backend needed):
python start_dashboard.py --mock
```

---

### Alternative: Running in Separate Terminals (Manual)

### Step 1: Start Live Inference & Forward-Testing Engine (Port 8000)
```bash
python -m backend.live.run --mode live
```
*(If developing with uncommitted changes, add `--allow-dirty`)*

### Step 2: Start Dashboard API Service (Port 8001)
```bash
uvicorn dashboard_api.app:app --port 8001
```

### Step 3: Serve the Frontend & Open the Dashboard
```bash
python -m http.server 8080 --directory frontend
```
Open in browser:
- **Connected Mode**: `http://localhost:8080/blank.html`
- **Offline / Demo Mode**: `http://localhost:8080/blank.html?mock=1` (works without any backend running using synthetic candles and simulated inferences)

---

## 2. Band Formulas & Visual Heuristics

The upper cap, lower cap, and smart price lines rendered on the chart are derived visual heuristics:

$$\text{ATR}_{14} = \text{Wilder's 14-period Smoothed Average True Range on closed candles}$$

$$\text{smart\_price} = \text{close}_t + (P_{\text{up}} - P_{\text{down}}) \times \text{ATR}_{14} \times \sqrt{2} \times K_{\text{MOVE}}$$

$$\text{half\_width} = Z \times \text{ATR}_{14} \times \sqrt{2} \times \left(0.75 + 0.5 \times (1 - P_{\text{neutral}})\right)$$

$$\text{upper} = \text{smart\_price} + \text{half\_width}$$
$$\text{lower} = \text{smart\_price} - \text{half\_width}$$

- **Defaults**: $K_{\text{MOVE}} = 0.5$, $Z = 1.28$.
- **Interpretation**: These bands are an *indicative range*, not a statistically calibrated prediction interval. They visualize projected price variance scaled by model conviction and current market volatility.
- **Containment Metric**: The UI tracks the empirical percentage of resolved trades whose candle $k+2$ close fell between the upper and lower caps.

---

## 3. Resolution & Verdict Rule (Mirroring `ledger.py`)

A trade signal formed at the close of candle $k$ resolves exactly 10 minutes later when candle $k+2$ closes. 

$$\text{fwd\_ret} = \frac{\text{close}_{k+2} - \text{close}_k}{\text{close}_k}$$

$$\text{raw\_sign}_2 = \begin{cases} +1.0 & \text{if } \text{fwd\_ret} > 0 \\ -1.0 & \text{if } \text{fwd\_ret} < 0 \\ 0.0 & \text{if } \text{fwd\_ret} == 0 \end{cases}$$

- **Directional Calls (`UP` / `DOWN`)**:
  - If $\text{raw\_sign}_2 == 0.0$ (exact zero move): marked `VOID` (neutralized; not counted in accuracy).
  - If $\text{action} == \text{raw\_sign}_2$: marked `✓ CORRECT` (Win).
  - If $\text{action} \neq \text{raw\_sign}_2$: marked `✗ WRONG` (Loss).
- **Neutral / No-Signal Calls**:
  - Marked `— NO CALL` (excluded from directional accuracy calculation).

---

## 4. Vocabulary Mapping

| Backend Internal Term | UI Display Term | Visual Badge |
| :--- | :--- | :--- |
| `LONG` | `UP` | Green Badge (`.badge.up`) |
| `SHORT` | `DOWN` | Red Badge (`.badge.down`) |
| `NEUTRAL` | `NO CLEAR SIGNAL` | Grey Badge (`.badge.neutral`) |
| `NO_SIGNAL` | `NO SIGNAL: <reason>` | Grey Badge with status reason |
| `BULLISH` | `UPTREND` | Green Trend Badge |
| `BEARISH` | `DOWNTREND` | Red Trend Badge |
| `SIDEWAYS` | `SIDEWAYS` | Grey Trend Badge |
| `IMPULSE` | `BREAKTHROUGH` | Green Phase Badge |
| `PULLBACK` | `PULLBACK` | Red/Orange Phase Badge |

---

## 5. Trader's Guide & System Manual

- **1-Second Signal Reading**: Check the large **FINAL PREDICTION** card. Green `UP` indicates expected upward expansion beyond transaction costs; Red `DOWN` indicates downward expansion; Grey `NO CLEAR SIGNAL` indicates consolidation where friction eliminates expected edge.
- **Confidence**: Model probabilities ($\max(P_{\text{up}}, P_{\text{down}})$) represent internal classifier confidence and are **not calibrated** to empirical win rates.
- **Frequent No-Calls**: The 23 bps friction floor deliberately filters out ~45% of noise candles to protect capital.
- **Failure Modes / When NOT to Trade**:
  - Stale data banner active (>15s websocket tick lag).
  - High-impact macroeconomic news spikes or flash crashes.
  - Less than 5 minutes after a connection drop or server restart.
  - Pre-inference feature imputation warnings (`imputed features`).

