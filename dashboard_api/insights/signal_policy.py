"""Display-layer signal policy and abstention logic (F1).

Pure, deterministic function computing display-layer action (shown_action).
Applies display-layer abstention without altering frozen model behavior.
"""

from typing import Tuple, Optional


# Frozen constants chosen on OOF train folds 1-3
DELTA_DEFAULT = 0.015
MARGIN_DEFAULT = 0.015


def compute_shown_action(
    ledger_action: str,
    p_up: float,
    p_down: float,
    p_neutral: float,
    status: str,
    delta: float = DELTA_DEFAULT,
    m: float = MARGIN_DEFAULT,
) -> Tuple[str, Optional[str]]:
    """Compute shown_action according to display-layer abstention policy.

    Parameters
    ----------
    ledger_action : str
        Verbatim ledger action label ('UP', 'DOWN', 'NEUTRAL', 'NO_SIGNAL', etc.)
    p_up : float
        Blended final probability for UP.
    p_down : float
        Blended final probability for DOWN.
    p_neutral : float
        Blended final probability for NEUTRAL.
    status : str
        Prediction status from ledger ('OK', 'RESOLVED', 'PENDING', 'NO_SIGNAL', etc.)
    delta : float
        Minimum excess plurality probability over uniform prior (1/3).
    m : float
        Minimum directional margin (|p_up - p_down|).

    Returns
    -------
    Tuple[str, Optional[str]]
        (shown_action, abstention_reason)
        shown_action is 'NO CLEAR SIGNAL' if abstained, otherwise the ledger action.
    """
    # 1. Non-OK / NO_SIGNAL check
    status_str = str(status).upper() if status else ""
    if "NO_SIGNAL" in status_str or "NO SIGNAL" in status_str or ledger_action == "NO_SIGNAL":
        return ("NO CLEAR SIGNAL", "status_not_ok")

    # 2. Plurality excess check: max(p) - 1/3 >= delta
    max_p = max(p_up, p_down, p_neutral)
    if (max_p - (1.0 / 3.0)) < delta:
        return ("NO CLEAR SIGNAL", "max_p_below_delta")

    # 3. Directional margin check: |p_up - p_down| >= m
    margin = abs(p_up - p_down)
    if margin < m:
        return ("NO CLEAR SIGNAL", "margin_below_m")

    # 4. Otherwise, pass through the ledger action
    # Map LONG -> UP, SHORT -> DOWN if necessary
    norm_action = ledger_action
    if norm_action == "LONG":
        norm_action = "UP"
    elif norm_action == "SHORT":
        norm_action = "DOWN"
    elif norm_action == "NEUTRAL":
        norm_action = "NEUTRAL"

    return (norm_action, None)
