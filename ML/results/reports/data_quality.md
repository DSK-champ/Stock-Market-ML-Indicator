# Data Quality & Integrity Report

**Date Generated:** 2026-11-10 07:22:22 UTC
**Target Symbol:** `BTCUSDT` | **Timeframe:** `5m`
**Expected Range:** `2023-11-01 00:00:00+00:00` to `2026-11-10 00:00:00+00:00`

---

## 1. Executive Summary

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Observed Candles** | `318,240` | - |
| **Expected Candles** | `318,240` | - |
| **Data Completeness** | `100.0000%` | PASS |
| **Identified Gaps** | `0` gaps (`0` missing candles) | PASS |
| **OHLC Consistency** | `0 violations` | PASS |
| **Non-Negative Volumes** | `0 negative rows` | PASS |
| **Zero-Volume Candles** | `0` candles | PASS |
| **Flagged Return Anomalies** | `1718` extreme candles | INFO (Audited) |

---

## 2. Structured Time Gaps Analysis

No time gaps detected. Data forms a perfectly continuous 5-minute grid.

---

## 3. OHLC & Volume Sanity Checks

| Check Description | Formula | Violation Count |
| :--- | :--- | :--- |
| Positive Prices | min(O, H, L, C) > 0 | `0` |
| High Upper Bound | H >= max(O, C) | `0` |
| Low Lower Bound | L <= min(O, C) | `0` |
| High vs Low Order | H >= L | `0` |
| Taker Volume Ceiling | Taker Buy <= Total Vol | `0` |

---

## 4. Extreme Return Anomalies (Robust 24h Rolling Z > 6.0 or |Return| > 5%)

Real market crashes and short squeezes are legitimate empirical data and are **retained** in the dataset. They are documented here for transparency.

Total extreme candles flagged: **1718**

| Timestamp (UTC) | Close Price | Single-Candle Return | Robust Z-Score | Reason |
| :--- | :--- | :--- | :--- | :--- |
| `2025-10-10 21:15:00+00:00` | $103,975.26 | `-7.62%` | `82.06` | Absolute Return > 5% |
| `2024-01-03 12:05:00+00:00` | $40,887.99 | `-4.95%` | `35.92` | Robust Rolling Z > 6.0 |
| `2024-12-05 22:25:00+00:00` | $94,115.90 | `-4.15%` | `22.08` | Robust Rolling Z > 6.0 |
| `2025-10-10 21:20:00+00:00` | $108,219.08 | `+4.00%` | `42.56` | Robust Rolling Z > 6.0 |
| `2026-03-23 11:05:00+00:00` | $71,216.41 | `+3.78%` | `32.44` | Robust Rolling Z > 6.0 |
| `2024-04-13 20:05:00+00:00` | $61,742.47 | `-3.78%` | `23.73` | Robust Rolling Z > 6.0 |
| `2024-08-05 14:00:00+00:00` | $53,618.02 | `+3.16%` | `8.43` | Robust Rolling Z > 6.0 |
| `2024-08-05 06:20:00+00:00` | $49,338.49 | `-3.10%` | `14.47` | Robust Rolling Z > 6.0 |
| `2024-08-05 13:30:00+00:00` | $51,273.99 | `+3.05%` | `8.47` | Robust Rolling Z > 6.0 |
| `2024-01-09 21:15:00+00:00` | $46,522.02 | `-2.96%` | `21.96` | Robust Rolling Z > 6.0 |
| `2024-03-04 00:45:00+00:00` | $62,404.92 | `-2.76%` | `28.07` | Robust Rolling Z > 6.0 |
| `2026-06-07 22:10:00+00:00` | $63,586.98 | `+2.73%` | `20.87` | Robust Rolling Z > 6.0 |
| `2025-10-10 21:30:00+00:00` | $111,338.21 | `+2.64%` | `27.81` | Robust Rolling Z > 6.0 |
| `2024-09-04 01:00:00+00:00` | $55,657.13 | `-2.63%` | `23.59` | Robust Rolling Z > 6.0 |
| `2024-08-08 01:40:00+00:00` | $57,346.54 | `+2.62%` | `13.08` | Robust Rolling Z > 6.0 |
