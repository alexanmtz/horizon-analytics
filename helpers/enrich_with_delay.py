import pandas as pd

def enrich_with_delay(df, mapping):
    arrival_col = mapping.get("arrival_at")
    expected_col = mapping.get("expected_arrival_at")

    if not arrival_col or not expected_col:
        return df  # can't compute delay yet

    out = df.copy()

    out[arrival_col] = pd.to_datetime(out[arrival_col], errors="coerce", utc=True)
    out[expected_col] = pd.to_datetime(out[expected_col], errors="coerce", utc=True)

    # delay in days (can be negative if early)
    out["delay_days"] = (out[arrival_col] - out[expected_col]).dt.total_seconds() / 86400.0

    # also handy:
    out["is_delayed"] = out["delay_days"] > 0
    out["delay_hours"] = out["delay_days"] * 24.0

    return out