# System Failures, Architectural Vulnerabilities & Edge Case Documentation

This document records algorithmic edge cases, tie-breaking mechanisms, and architectural failure modes audited during diagnostic inspection.

---

## 1. `np.argmax` Tie-Break Asymmetry to SHORT

### Mechanism (`ML/src/trading_engine.py:284`, `backend/live/api.py:180`)
In the hierarchical decision logic and fallback handlers:
```python
probs_vector = [p_down, p_neutral, p_up]
argmax_idx = int(np.argmax(probs_vector))
```
- By default, Python/NumPy `np.argmax` returns the **first index** achieving the maximum value.
- When an exact uninformative uniform prior or exact tie occurs between all classes:
  $$(p_{\text{down}}, p_{\text{neutral}}, p_{\text{up}}) = \left(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right)$$
  `np.argmax` resolves to index `0`, corresponding to `SHORT` (DOWN).
- Similarly, in `trading_engine.py:280`:
  ```python
  if u >= d and u >= n:
      action = 1
  elif d >= u and d >= n:
      action = -1
  ```
  If $u = d = n = 1/3$, the non-strict inequality `u >= d and u >= n` is satisfied ($1/3 \ge 1/3$), triggering `action = 1` (LONG). But if $u < n$ and $d = u \ge \theta_{\text{dir}}$, `d > u` is false, so it falls to neutral `0`.

### OOF Empirical Audit
- Total evaluated Out-of-Fold (OOF) rows: **110,433**
- Exact ties between $p_{\text{up}}$ and $p_{\text{down}}$ ($p_{\text{up}} == p_{\text{down}}$): **0 / 110,433 (0.00%)**
- Exact ties between all three classes ($p_{\text{up}} == p_{\text{neutral}} == p_{\text{down}}$): **0 / 110,433 (0.00%)**
- **Conclusion:** While continuous floating-point probability outputs make exact ties measure zero on historical data, uninformative fallback vectors (e.g. `(1/3, 1/3, 1/3)`) produce asymmetric directional bias.

---

## 2. Absence of Absolute Probability Floor in Directional Neutral Override

### Mechanism (`ML/src/trading_engine.py:286-291`)
```python
# Directional Neutral Override:
# When NEUTRAL has plurality (n >= u and n >= d), but a directional move is credible
if max(u, d) >= self.theta_dir and abs(u - d) >= self.margin:
    action = 1 if u > d else (-1 if d > u else 0)
```
- Constants: $\theta_{\text{dir}} = 0.32$, $\text{margin} = 0.04$.
- Because $\theta_{\text{dir}} = 0.32 < \frac{1}{3} \approx 0.3333$, this directional override branch lacks a probability floor exceeding uniform chance.
- A near-uniform vector such as:
  $$(p_{\text{down}}, p_{\text{neutral}}, p_{\text{up}}) = (0.31, 0.34, 0.35)$$
  has $u = 0.35 \ge 0.32$ and $|u - d| = |0.35 - 0.31| = 0.04 \ge 0.04$.
  Even though NEUTRAL has plurality ($p_{\text{neutral}} = 0.34$) and $p_{\text{up}} = 0.35$ is barely above random uniform guessing, the rule overrides NEUTRAL and issues an active directional call (`action = 1`, LONG).
- **Remediation:** Display-layer abstention filter (`F1: shown_action`) suppresses these near-uniform calls in the user interface without altering the frozen model weights.
