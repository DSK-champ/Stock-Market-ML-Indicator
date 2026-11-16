"""Display-layer signal policy and abstention logic (F1).

Pure, deterministic function computing display-layer action (shown_action)
and derived display_state at read time.
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
        Minimum directional margin (|p_up - p_down|). Applies only to UP and DOWN.

    Returns
    -------
    Tuple[str, Optional[str]]
        (shown_action, abstention_reason)
        shown_action is 'NO CLEAR SIGNAL' if abstained, otherwise the ledger action.
    """
    # 1. Non-OK / NO_SIGNAL check
    status_str = str(status).upper() if status else ""
    raw_str = str(ledger_action).upper() if ledger_action else ""
    if "NO_SIGNAL" in status_str or "NO SIGNAL" in status_str or raw_str == "NO_SIGNAL":
        return ("NO CLEAR SIGNAL", "status_not_ok")

    # 2. Plurality excess check: max(p) - 1/3 >= delta (applies to all rows)
    max_p = max(p_up, p_down, p_neutral)
    if (max_p - (1.0 / 3.0)) < delta:
        return ("NO CLEAR SIGNAL", "below_delta_threshold")

    # Normalize action
    norm_action = raw_str
    if norm_action in ("LONG", "1"):
        norm_action = "UP"
    elif norm_action in ("SHORT", "-1"):
        norm_action = "DOWN"
    elif norm_action in ("NEUTRAL", "0"):
        norm_action = "NEUTRAL"

    # 3. Directional margin check: |p_up - p_down| >= m (applies ONLY to UP and DOWN)
    if norm_action in ("UP", "DOWN"):
        margin = abs(p_up - p_down)
        if margin < m:
            return ("NO CLEAR SIGNAL", "margin_violation")

    # 4. Otherwise, pass through the ledger action (UP, DOWN, or NEUTRAL)
    return (norm_action, None)


def compute_display_state(
    raw_action: str,
    p_up: float,
    p_down: float,
    p_neutral: float,
    status: Optional[str] = None,
    status_reason: Optional[str] = None,
    delta: float = DELTA_DEFAULT,
    m: float = MARGIN_DEFAULT,
) -> str:
    """Compute derived display_state at read time without modifying stored data.

    States
    ------
    - NO SIGNAL: <reason> (data problem)
    - NO CLEAR SIGNAL (weak probabilities max(p)-1/3 < delta, or directional lean failing margin)
    - NEUTRAL (ledger NEUTRAL and max(p) - 1/3 >= delta)
    - UP, DOWN (calls passing both plurality and margin tests)
    """
    status_str = str(status) if status is not None else ""
    raw_str = str(raw_action).upper() if raw_action is not None else ""

    # 1. Data problem
    if status_str.startswith("NO SIGNAL:"):
        return status_str
    if "NO_SIGNAL" in status_str or "NO SIGNAL" in status_str or raw_str == "NO_SIGNAL":
        reason = status_reason or (status_str.split(":", 1)[1].strip() if ":" in status_str else "data problem")
        return f"NO SIGNAL: {reason}"

    # 2. Weak-probability test (max(p) - 1/3 < delta) applies to all rows
    max_p = max(p_up, p_down, p_neutral)
    if (max_p - (1.0 / 3.0)) < delta:
        return "NO CLEAR SIGNAL"

    # Normalize action
    norm_act = raw_str
    if norm_act in ("LONG", "1"):
        norm_act = "UP"
    elif norm_act in ("SHORT", "-1"):
        norm_act = "DOWN"
    elif norm_act in ("NEUTRAL", "0"):
        norm_act = "NEUTRAL"

    # 3. Margin test applies only when ledger action is UP or DOWN
    if norm_act in ("UP", "DOWN"):
        if abs(p_up - p_down) < m:
            return "NO CLEAR SIGNAL"
        return norm_act

    # 4. Ledger NEUTRAL (and max(p) - 1/3 >= delta)
    return "NEUTRAL"

