import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "quote_volume",
    "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
]

PRICE_COLUMNS = ["open", "high", "low", "close"]

VOLUME_COLUMNS = [
    "volume", "quote_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume",
]

NUMERIC_COLUMNS = PRICE_COLUMNS + VOLUME_COLUMNS


# ============================================================
#                 VALIDATE DATAFRAME STRUCTURE
# ============================================================

def validate_columns(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    return {
        "valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
    }


# ============================================================
#              VALIDATE INDIVIDUAL CANDLE VALUES
# ============================================================

def validate_candle(candle):

    errors = []

    # Check required fields
    missing_fields = [
        column for column in REQUIRED_COLUMNS
        if column not in candle
    ]

    if missing_fields:
        return {
            "valid": False,
            "errors": [f"Missing fields: {missing_fields}"],
        }

    # Check timestamp
    try:
        timestamp = pd.Timestamp(candle["open_time"])

        if pd.isna(timestamp):
            errors.append("open_time is missing or invalid.")

    except (TypeError, ValueError, OverflowError):
        errors.append("open_time is invalid.")

    # Check numeric values are finite
    for column in NUMERIC_COLUMNS:
        try:
            value = float(candle[column])

            if not np.isfinite(value):
                errors.append(f"{column} must be finite.")

        except (TypeError, ValueError, OverflowError):
            errors.append(f"{column} must be numeric.")
            
    # Further checks require valid numeric values
    try:
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price  = float(candle["low"])
        close_price = float(candle["close"])

        prices_valid = all(
            np.isfinite(value) and value > 0
            for value in [
                open_price, high_price, low_price, close_price
            ]
        )

        if prices_valid:
            # Check OHLC consistency
            if high_price < max(open_price, close_price, low_price):
                errors.append("high is inconsistent with OHLC prices.")

            if low_price > min(open_price, close_price, high_price):
                errors.append("low is inconsistent with OHLC prices.")

        elif all(np.isfinite(value) for value in [
            open_price, high_price, low_price, close_price
        ]):
            errors.append("OHLC prices must be positive.")

    except (TypeError, ValueError, OverflowError):
        pass

    # Check non-negative volume and trade values
    for column in VOLUME_COLUMNS:
        try:
            value = float(candle[column])

            if np.isfinite(value) and value < 0:
                errors.append(f"{column} cannot be negative.")

        except (TypeError, ValueError, OverflowError):
            pass

    # Taker-buy volume cannot exceed total volume
    try:
        volume = float(candle["volume"])
        taker_buy_volume = float(candle["taker_buy_base_volume"])

        if (
            np.isfinite(volume)
            and np.isfinite(taker_buy_volume)
            and volume >= 0
            and taker_buy_volume > volume
        ):
            errors.append(
                "taker_buy_base_volume exceeds total volume."
            )

    except (TypeError, ValueError, OverflowError):
        pass

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


# ============================================================
#                  VALIDATE DATAFRAME VALUES
# ============================================================

def validate_values(df):

    structure = validate_columns(df)

    if not structure["valid"]:
        return {
            "valid": False,
            "missing_columns": structure["missing_columns"],
            "invalid_rows": [],
        }

    invalid_rows = []

    # Validate each candle individually
    for index, row in df.iterrows():

        result = validate_candle(row)

        if not result["valid"]:
            invalid_rows.append({
                "index": index,
                "open_time": row.get("open_time"),
                "errors": result["errors"],
            })

    return {
        "valid": len(invalid_rows) == 0,
        "missing_columns": [],
        "invalid_rows": invalid_rows,
    }


# ============================================================
#             VALIDATE TIMESTAMPS AND CHRONOLOGY
# ============================================================

def validate_timestamps(df, interval="5min"):

    structure = validate_columns(df)

    if not structure["valid"]:
        return {
            "valid": False,
            "error": f"Missing columns: {structure['missing_columns']}",
        }

    timestamps = pd.to_datetime(df["open_time"], utc=True, errors="coerce",)

    invalid_timestamps = int(timestamps.isna().sum())
    duplicate_timestamps = int(timestamps.duplicated().sum())
    chronological = bool(timestamps.is_monotonic_increasing)

    interval_delta = pd.Timedelta(interval)
    intervals = timestamps.diff()

    invalid_intervals = int(
        (intervals.notna() & (intervals != interval_delta)).sum()
    )

    return {
        "valid": (
            invalid_timestamps == 0
            and duplicate_timestamps == 0
            and chronological
            and invalid_intervals == 0
        ),
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "chronological": chronological,
        "invalid_intervals": invalid_intervals,
    }


# ============================================================
#             VALIDATE MISSING AND INFINITE VALUES
# ============================================================

def validate_missing_values(df):

    structure = validate_columns(df)

    if not structure["valid"]:
        return {
            "valid": False,
            "missing_columns": structure["missing_columns"],
        }

    missing_values = df[REQUIRED_COLUMNS].isna().sum().to_dict()

    infinite_values = {}

    for column in NUMERIC_COLUMNS:
        numeric_values = pd.to_numeric(df[column], errors="coerce",)
        infinite_values[column] = int(np.isinf(numeric_values).sum())

    return {
        "valid": (
            sum(missing_values.values()) == 0
            and sum(infinite_values.values()) == 0
        ),
        "missing_values": missing_values,
        "infinite_values": infinite_values,
    }


# ============================================================
#                RUN COMPLETE DATASET VALIDATION
# ============================================================

def validate_dataframe(df, interval="5min"):

    structure = validate_columns(df)

    if not structure["valid"]:
        return {
            "valid": False,
            "missing_columns": structure["missing_columns"],
        }

    values_report = validate_values(df)
    timestamps_report = validate_timestamps(df, interval)
    missing_report = validate_missing_values(df)

    return {
        "valid": (
            values_report["valid"]
            and timestamps_report["valid"]
            and missing_report["valid"]
        ),
        "row_count": len(df),
        "values": values_report,
        "timestamps": timestamps_report,
        "missing_data": missing_report,
    }


# ============================================================
#                    PRINT VALIDATION REPORT
# ============================================================

def print_validation_report(report):

    print("=" * 55)
    print("DATA VALIDATION REPORT")
    print("=" * 55)

    if "row_count" in report: print(f"Total rows: {report['row_count']:,}")
    print(f"Dataset valid: {report['valid']}")

    if "missing_columns" in report:
        print("Missing columns:", report["missing_columns"])
        return

    values = report["values"]
    timestamps = report["timestamps"]
    missing = report["missing_data"]

    print("\nVALUE CHECKS")
    print("-" * 55)
    print("Invalid candles:", len(values["invalid_rows"]))

    print("\nTIMESTAMP CHECKS")
    print("-" * 55)
    print("Invalid timestamps:", timestamps["invalid_timestamps"])
    print("Duplicate timestamps:", timestamps["duplicate_timestamps"])
    print("Chronologically sorted:", timestamps["chronological"])
    print("Unexpected intervals:", timestamps["invalid_intervals"])

    print("\nMISSING / INFINITE VALUE CHECKS")
    print("-" * 55)
    print("Missing values:", sum(missing["missing_values"].values()))
    print("Infinite values:", sum(missing["infinite_values"].values()))

    print("=" * 55)
