/**
 * Model-Informed Trade Zones Configuration
 * Central client-side constants and multipliers.
 */

window.ZONE_CONFIG = {
  k_entry: 0.75,
  k_stop: 1.25,
  scale: 1.50,
  buffer_bps: 5.0,
  cost_floor_bps: 23.0,
  theta_dir: 0.32,
  high_vol_multiplier: 1.35,
  horizon_candles: 2,
  high_vol_atr_threshold: 0.0070,
  swing_lookback: 12,
  
  // UI Display state
  show_on_chart: true,
  active_only: true,
};
