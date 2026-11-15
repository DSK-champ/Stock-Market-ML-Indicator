"""Insights module for display-layer policy, abstention, empirical bands, and telemetry."""

from dashboard_api.insights.signal_policy import compute_shown_action, DELTA_DEFAULT, MARGIN_DEFAULT
from dashboard_api.insights.store import InsightsStore
from dashboard_api.insights.empirical_band import compute_empirical_band, K_SMART_PRICE

__all__ = [
    "compute_shown_action",
    "DELTA_DEFAULT",
    "MARGIN_DEFAULT",
    "InsightsStore",
    "compute_empirical_band",
    "K_SMART_PRICE",
]
