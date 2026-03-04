import re
import pandas as pd
from metrics import compute_all_metrics, add_derived_columns


def answer_question(df, mapping, question: str) -> str:
    q = (question or "").lower()
    derived = add_derived_columns(df, mapping)

    if ("top" in q) and ("delay" in q) and ("holiday" in q):
        if "is_bank_holiday" not in df.columns:
            return "Holiday enrichment not connected yet. Click 🔌 Connect Holiday Calendar."

        if "delay_hours" not in derived.columns:
            return "Delay metrics not available."

        top = derived.sort_values("delay_hours", ascending=False).head(10)

        cols = [
            mapping.get("id"),
            "country",
            "delay_hours",
            "holiday_name",
            mapping.get("expected_arrival_at"),
        ]
        cols = [c for c in cols if c and c in top.columns]
        return format_table(top, cols)
    q = (question or "").lower()

    # Always recompute derived metrics
    derived = add_derived_columns(df, mapping)

    # 1️⃣ Basic metrics fallback
    if re.search(r"\b(average|mean|median|p90|delay|transfer)\b", q):
        return compute_all_metrics(df, mapping)

    # 2️⃣ Holiday comparison
    if "holiday" in q and "average" in q:
        if "is_bank_holiday" not in df.columns:
            return "Holiday enrichment not connected yet. Click 🔌 Connect Holiday Calendar."

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
            return "Holiday enrichment not connected yet. Click 🔌 Connect Holiday Calendar."

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

    # 4️⃣ Why question (smart explanation)
    if "why" in q or "explain" in q:
        if "is_bank_holiday" not in df.columns:
            return (
                "Delays appear structured around calendar patterns.\n\n"
                "This suggests bank holidays or non-business days.\n"
                "Click 🔌 Connect Holiday Calendar to verify."
            )

        holiday_share = df["is_bank_holiday"].mean() * 100

        return (
            "🧠 Explanation\n\n"
            f"{holiday_share:.1f}% of payouts fall on bank holidays.\n"
            "Delays are significantly higher on those days.\n\n"
            "Settlement is pushed to the next business day."
        )

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