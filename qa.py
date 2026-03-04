import re
import pandas as pd
from metrics import compute_all_metrics, add_derived_columns


def answer_question(df, mapping, question: str) -> str:
    q = (question or "").lower()
    derived = add_derived_columns(df, mapping)

    # 1️⃣ Basic metrics fallback
    if re.search(r"\b(average|mean|median|p90|delay|transfer)\b", q):
        return compute_all_metrics(df, mapping)

    # 2️⃣ Holiday comparison
    if "holiday" in q and "average" in q:
        if "is_bank_holiday" not in df.columns:
            return "Holiday enrichment is not connected yet. Ask me to suggest an external calendar datasource."

        if "delay_hours" not in derived.columns:
            return "Delay metrics not available."

        holiday_avg = derived.loc[
            derived["is_bank_holiday"] & derived["delay_hours"].notna(),
            "delay_hours"
        ].mean()

        non_avg = derived.loc[
            (~derived["is_bank_holiday"]) & derived["delay_hours"].notna(),
            "delay_hours"
        ].mean()

        return (
            "📊 Holiday vs Non-Holiday Delay Comparison\n\n"
            f"- Avg delay on holidays: **{holiday_avg:.2f}h**\n"
            f"- Avg delay on non-holidays: **{non_avg:.2f}h**\n"
            f"- Ratio: **{holiday_avg/non_avg:.2f}x slower on holidays**"
        )

    # 3️⃣ Show top delays with holiday names
    if "top" in q and "holiday" in q:
        if "is_bank_holiday" not in df.columns:
            return "Holiday enrichment is not connected yet. Ask me to suggest an external calendar datasource."

        top = derived.sort_values("delay_hours", ascending=False).head(10)

        cols = [
            mapping.get("id"),
            "country",
            "delay_hours",
            "holiday_name",
            mapping.get("expected_arrival_at"),
        ]

        cols = [c for c in cols if c in top.columns]

        return format_table(top, cols)

    # 3.5️⃣ Explain holiday impact directly from enriched columns
    if "holiday" in q and ("impact" in q or "explain" in q):
        if "is_bank_holiday" not in df.columns:
            return "Holiday enrichment is not connected yet. Ask me to suggest an external calendar datasource."

        if "delay_hours" not in derived.columns:
            return "Delay metrics not available."

        holiday_delay = derived.loc[
            derived["is_bank_holiday"] & derived["delay_hours"].notna(),
            "delay_hours"
        ]
        non_holiday_delay = derived.loc[
            (~derived["is_bank_holiday"]) & derived["delay_hours"].notna(),
            "delay_hours"
        ]

        holiday_count = int(holiday_delay.shape[0])
        non_holiday_count = int(non_holiday_delay.shape[0])

        if holiday_count == 0:
            return (
                "Holiday enrichment is connected, but there are no payout rows matched to holidays "
                "in the current dataset/date range."
            )

        holiday_avg = float(holiday_delay.mean()) if holiday_count > 0 else float("nan")
        non_avg = float(non_holiday_delay.mean()) if non_holiday_count > 0 else float("nan")

        ratio_text = "N/A"
        if pd.notna(non_avg) and non_avg > 0:
            ratio_text = f"{holiday_avg / non_avg:.2f}x"

        findings = [
            "🧠 Explanation",
            "",
            f"- Holiday payouts analyzed: **{holiday_count}**",
            f"- Non-holiday payouts analyzed: **{non_holiday_count}**",
            f"- Avg delay on holidays: **{holiday_avg:.2f}h**",
            f"- Avg delay on non-holidays: **{non_avg:.2f}h**" if pd.notna(non_avg) else "- Avg delay on non-holidays: **N/A**",
            f"- Relative holiday impact: **{ratio_text}**",
        ]

        if "holiday_name" in derived.columns:
            top_holidays = (
                derived.loc[derived["is_bank_holiday"] & derived["holiday_name"].astype(str).str.len().gt(0)]
                .groupby("holiday_name")["delay_hours"]
                .agg(["mean", "count"])
                .sort_values(["mean", "count"], ascending=[False, False])
                .head(3)
            )
            if not top_holidays.empty:
                findings.append("- Highest-delay holiday names in this data:")
                for name, row in top_holidays.iterrows():
                    findings.append(f"  - {name}: {row['mean']:.2f}h (n={int(row['count'])})")

        return "\n".join(findings)

    # 4️⃣ Why question (smart explanation)
    if "why" in q or "explain" in q:
        return _reason_discovery_summary(derived, mapping)

    # Fallback
    return compute_all_metrics(df, mapping)


def format_table(df: pd.DataFrame, cols: list[str], max_rows: int = 10) -> str:
    view = df[cols].head(max_rows).copy()

    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].round(2)

    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"

    rows = []
    for _, r in view.iterrows():
        rows.append("| " + " | ".join([str(r[c]) for c in view.columns]) + " |")

    return "\n".join([header, sep] + rows)


def _reason_discovery_summary(derived: pd.DataFrame, mapping: dict) -> str:
    if "delay_hours" not in derived.columns or derived["delay_hours"].notna().sum() == 0:
        return "I can't explain delay drivers yet because delay metrics are unavailable. Check timestamp mapping first."

    delay = derived["delay_hours"].dropna()
    baseline = delay.mean()
    findings = []

    candidate_dims = [
        "country",
        "bank",
        "bank_name",
        "provider",
        mapping.get("currency"),
        mapping.get("status"),
    ]
    candidate_dims = [c for c in candidate_dims if c and c in derived.columns]

    strongest_dim = None
    strongest_effect = 0.0

    for dim in candidate_dims:
        grp = derived.dropna(subset=[dim, "delay_hours"]).groupby(dim)["delay_hours"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 5]
        if grp.empty:
            continue

        span = float(grp["mean"].max() - grp["mean"].min())
        if span > strongest_effect:
            strongest_effect = span
            strongest_dim = dim

    if strongest_dim is not None:
        grp = derived.dropna(subset=[strongest_dim, "delay_hours"]).groupby(strongest_dim)["delay_hours"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 5].sort_values("mean", ascending=False)
        top_label = str(grp.index[0])
        top_mean = float(grp.iloc[0]["mean"])
        findings.append(f"Largest internal driver appears to be `{strongest_dim}` (top segment `{top_label}` avg `{top_mean:.2f}h`).")

    if "arrival_weekday" in derived.columns:
        w = derived.dropna(subset=["arrival_weekday", "delay_hours"]).groupby("arrival_weekday")["delay_hours"].mean()
        if not w.empty:
            peak_day = str(w.idxmax())
            peak_mean = float(w.max())
            findings.append(f"Delay also peaks on `{peak_day}` (`{peak_mean:.2f}h` average).")

    residual_hint = strongest_effect < max(2.0, baseline * 0.25)
    if residual_hint:
        findings.append("Internal columns only weakly explain delay variance; external calendar or operations context may help.")

    if not findings:
        return "I checked internal segments but found no strong delay drivers yet."

    return "🧠 Explanation\n\n- " + "\n- ".join(findings)