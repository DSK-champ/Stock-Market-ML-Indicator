# Formal Project Sources & Citations

Every assumption, threshold, fee rate, paper, and library behavior referenced in this project is documented below with its verification status.

---

## 1. Exchange Rules & API Limits

### Binance Spot VIP 0 Trading Fee Schedule
- **Status:** `[VERIFIED]` (Checked November 2026)
- **Maker Fee:** 0.1000%
- **Taker Fee:** 0.1000%
- **Round-Trip Fee:** Standard VIP 0 taker entry + taker exit = $0.1000\% \times 2 = 0.2000\%$.
- **Source:** Binance Official Trading Fee Schedule  
  URL: `https://www.binance.com/en/fee/trading`

### BNB Fee Deduction (Documented Alternative)
- **Status:** `[VERIFIED ALTERNATIVE - DO NOT USE UNLESS DIRECTED]`
- **Terms:** 25% discount when trading fees are paid in BNB on Binance, lowering Maker and Taker fees to 0.0750% each (round-trip 0.1500%).
- **Policy:** Preserved as a documented alternative baseline only; not used as the default modeling friction unless specifically requested.

### Execution Slippage & Bid-Ask Spread
- **Status:** `[ASSUMPTION]` (Modeling friction estimates for liquid BTCUSDT 5m order book)
- **Estimated Slippage:** 0.0200% (0.0002) one-way adverse price impact on market orders.
- **Estimated Half-Spread:** 0.0100% (0.0001) effective half-spread crossing cost on BTCUSDT.
- **Total Round-Trip Economic Cost:**
  $$\text{Total Friction} = \text{Round-trip Fees (0.20\%)} + \text{Slippage (0.02\%)} + \text{Half-Spread (0.01\%)} = 0.2300\% \; (0.0023)$$
  This sum forms the exact economic basis for `LABEL_DEAD_ZONE = 0.0023`.

### Binance Kline/Candlestick REST API Specifications
- **Status:** `[VERIFIED]` (Checked November 2026)
- **Endpoint:** `GET /api/v3/klines` (and public vision mirror: `https://data-api.binance.vision/api/v3/klines`)
- **Request Weight:** 2 per request.
- **Default IP Weight Limit:** 1,200/min (legacy/base) up to 6,000/min infrastructure limit.
- **Max Response Rows:** 1,000 klines per single call.
- **Timestamp Unit:** Milliseconds (Unix epoch).
- **Source:** Binance Official Spot API Documentation  
  URL: `https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data`

---

## 2. Quantitative Finance & ML Literature

### Purged and Embargoed Walk-Forward Validation
- **Status:** `[VERIFIED]`
- **Concept:** Prevents information leakage caused by overlapping forward-return labels across chronological train/validation/test boundaries. Requires a purge gap of length equal to the label horizon $N$.
- **Citation:** López de Prado, Marcos (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. Chapter 7: Cross-Validation in Finance.

### Probability Calibration in Machine Learning
- **Status:** `[VERIFIED]`
- **Concept:** Uncalibrated tree ensemble / neural network probabilities frequently exhibit overconfidence or distortion. Post-processing with Platt scaling (logistic sigmoid) or Isotonic Regression maps raw model outputs to true empirical class probabilities.
- **Citation:** Niculescu-Mizil, Alexandru, and Rich Caruana (2005). *Predicting Good Probabilities With Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML).

### Market Microstructure & Friction-Aware Labeling
- **Status:** `[VERIFIED]`
- **Concept:** Return series at high frequencies (e.g., 5-minute candles) are dominated by bid-ask bounce and noise. Establishing an economic dead-zone equal to round-trip transaction costs ensures models predict tradeable signal rather than noise.
- **Citation:** Cartea, Álvaro, Sebastian Jaimungal, and José Penalva (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

### BlackRock Quantitative Factor Categorization
- **Status:** `[TO VERIFY]` (Pending exact document/link from mentor/user)
- **Concept:** Structuring alpha factors into orthogonal groupings:
  1. Trend & Momentum (SMA, EMA, RSI, MACD, ADX)
  2. Mean Reversion & Volatility (Rolling Std, Normalized ATR, Bollinger %B)
  3. Volume & Liquidity (Volume Z-score, Log Volume)
  4. Microstructure & Order Flow (Taker Buy Ratio, Trade Count Z-score)
- **Preliminary Citation:** Ang, Andrew (2014). *Asset Management: A Systematic Approach to Factor Investing*. Oxford University Press.

---

## 3. Structure, Pivots & Technical Architecture (Step 4b)

### Dow Theory & Algorithmic Trend State Machine
- **Status:** `[VERIFIED]`
- **Concept:** Market structure is partitioned into directional states (Up, Down, Range) based on sequential swing extremes. An uptrend persists as long as price produces consecutive Higher Highs (HH) and Higher Lows (HL); a close breaking below the last confirmed HL by a buffer terminates the uptrend into Range or Down.
- **Citation:** Rhea, Robert (1932). *The Dow Theory*. Barron's. Edwards, Robert D., and John Magee (1948). *Technical Analysis of Stock Trends*. John Magee Inc.

### Fractal Swing Highs and Lows (Williams Fractals)
- **Status:** `[VERIFIED]`
- **Concept:** A local swing high at candle $i$ requires $H_i > \max(H_{i-k \dots i-1})$ and $H_i > \max(H_{i+1 \dots i+k})$. A local swing low requires $L_i < \min(L_{i-k \dots i-1})$ and $L_i < \min(L_{i+1 \dots i+k})$. In high-frequency causal modeling, pivot $i$ is strictly invisible prior to candle $i+k$ (the confirmation candle).
- **Citation:** Williams, Bill (1995). *Trading Chaos: Applying Expert Techniques to Maximize Your Profits*. John Wiley & Sons.

### Wilder's Average True Range (ATR)
- **Status:** `[VERIFIED]`
- **Concept:** Volatility metric accounting for gaps and extreme intraday price spans.
  $$TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$
  $$\text{ATR}_{t} = \frac{13 \cdot \text{ATR}_{t-1} + TR_t}{14}$$
  Normalized by closing price $\text{ATR}_t / C_t$ to provide a stationary, unitless measure of volatility.
- **Citation:** Wilder, J. Welles (1978). *New Concepts in Technical Trading Systems*. Trend Research.


