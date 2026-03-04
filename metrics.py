import pandas as pd
import numpy as np


def compute_all_metrics(df: pd.DataFrame, mapping: dict) -> str:
    derived = df.copy()

    lines = []
    lines.append("### 📌 Key metrics")

    if "transfer_time_hours" in derived.columns and derived["transfer_time_hours"].notna().any():
        tt = derived["transfer_time_hours"].dropna()
        lines.append(f"- Transfer time (hours): avg **{tt.mean():.2f}**, median **{tt.median():.2f}**, p90 **{np.percentile(tt, 90):.2f}**")
    else:
        lines.append("- Transfer time: **could not be calculated** (`paid_at` and/or `arrival_at` is missing).")

    if "delay_hours" in derived.columns and derived["delay_hours"].notna().any():
        dh = derived["delay_hours"].dropna()
        lines.append(f"- Delay (hours): avg **{dh.mean():.2f}**, median **{dh.median():.2f}**, p90 **{np.percentile(dh, 90):.2f}**")
        lines.append("")
        lines.append("### 🧨 Top 10 largest delays")
        top = derived.sort_values("delay_hours", ascending=False).head(10)
        cols_show = [c for c in [mapping.get("id"), "delay_hours", mapping.get("currency"), mapping.get("amount"), mapping.get("status")] if c]
        cols_show = [c for c in cols_show if c in top.columns]
        lines.append(format_table(top, cols_show))
    else:
        lines.append("- Delay: **could not be calculated** (`expected_arrival_at` and/or `arrival_at` is missing).")

    # Delay by weekday (if arrival exists)
    if "arrival_weekday" in derived.columns and "delay_hours" in derived.columns and derived["delay_hours"].notna().any():
        lines.append("")
        lines.append("### 📆 Delay by day of week (average)")
        grp = derived.dropna(subset=["delay_hours", "arrival_weekday"]).groupby("arrival_weekday")["delay_hours"].mean().reset_index()
        lines.append(format_table(grp, ["arrival_weekday", "delay_hours"]))

    return "\n".join(lines)


def add_derived_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    d = df.copy()

    def to_dt(colname: str):
        if not colname or colname not in d.columns:
            return None
        return pd.to_datetime(d[colname], errors="coerce", utc=True)

    paid = to_dt(mapping.get("paid_at"))
    arrival = to_dt(mapping.get("arrival_at"))
    expected = to_dt(mapping.get("expected_arrival_at"))

    if paid is not None and arrival is not None:
        d["transfer_time_hours"] = (arrival - paid).dt.total_seconds() / 3600.0

    if expected is not None and arrival is not None:
        d["delay_hours"] = (arrival - expected).dt.total_seconds() / 3600.0

    if arrival is not None:
        d["arrival_weekday"] = arrival.dt.day_name()

    return d


def format_table(df: pd.DataFrame, cols: list[str], max_rows: int = 10) -> str:
    # simple markdown table
    view = df[cols].head(max_rows).copy()
    # round floats
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].round(2)

    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = []
    for _, r in view.iterrows():
        rows.append("| " + " | ".join([safe_cell(r[c]) for c in view.columns]) + " |")
    return "\n".join([header, sep] + rows)


def safe_cell(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)