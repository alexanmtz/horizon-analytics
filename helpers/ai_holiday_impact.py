import pandas as pd

from clients.openai_client import client


def summarize_holiday_impact(df: pd.DataFrame) -> dict | None:
    required = {"delay_hours", "is_bank_holiday"}
    if not required.issubset(set(df.columns)):
        return None

    holiday_delay = df.loc[df["is_bank_holiday"] & df["delay_hours"].notna(), "delay_hours"]
    non_holiday_delay = df.loc[(~df["is_bank_holiday"]) & df["delay_hours"].notna(), "delay_hours"]

    holiday_count = int(holiday_delay.shape[0])
    non_holiday_count = int(non_holiday_delay.shape[0])

    holiday_avg = float(holiday_delay.mean()) if holiday_count > 0 else None
    non_holiday_avg = float(non_holiday_delay.mean()) if non_holiday_count > 0 else None

    ratio = None
    if holiday_avg is not None and non_holiday_avg is not None and non_holiday_avg > 0:
        ratio = holiday_avg / non_holiday_avg

    summary = {
        "holiday_count": holiday_count,
        "non_holiday_count": non_holiday_count,
        "holiday_avg": holiday_avg,
        "non_holiday_avg": non_holiday_avg,
        "ratio": ratio,
    }

    if "holiday_name" in df.columns:
        top_holiday = (
            df.loc[df["is_bank_holiday"] & df["holiday_name"].astype(str).str.len().gt(0)]
            .groupby("holiday_name")["delay_hours"]
            .agg(["mean", "count"])
            .sort_values(["mean", "count"], ascending=[False, False])
            .head(3)
            .reset_index()
            .to_dict(orient="records")
        )
        summary["top_holidays"] = top_holiday
    else:
        summary["top_holidays"] = []

    return summary


def holiday_impact_fallback_markdown(summary: dict) -> str:
    holiday_avg = summary.get("holiday_avg")
    non_holiday_avg = summary.get("non_holiday_avg")
    ratio = summary.get("ratio")

    holiday_avg_text = f"{holiday_avg:.2f}h" if holiday_avg is not None else "N/A"
    non_avg_text = f"{non_holiday_avg:.2f}h" if non_holiday_avg is not None else "N/A"
    ratio_text = f"{ratio:.2f}x" if ratio is not None else "N/A"

    lines = [
        "🧠 Holiday impact",
        "",
        f"- Holiday payouts analyzed: **{summary['holiday_count']}**",
        f"- Non-holiday payouts analyzed: **{summary['non_holiday_count']}**",
        f"- Avg holiday delay: **{holiday_avg_text}**",
        f"- Avg non-holiday delay: **{non_avg_text}**",
        f"- Relative impact: **{ratio_text}**",
    ]

    top_holidays = summary.get("top_holidays") or []
    if top_holidays:
        lines.append("- Highest-delay holiday names in this dataset:")
        for row in top_holidays:
            lines.append(f"  - {row['holiday_name']}: {row['mean']:.2f}h (n={int(row['count'])})")

    return "\n".join(lines)


async def ai_explain_holiday_impact(df: pd.DataFrame, mapping: dict) -> str:
    summary = summarize_holiday_impact(df)
    if summary is None:
        return "Holiday impact is unavailable because required columns (`is_bank_holiday`, `delay_hours`) are missing."

    if summary["holiday_count"] == 0:
        return (
            "Holiday enrichment is connected, but there are no payout rows matched to holidays in the current dataset/date range."
        )

    prompt = f"""
You are an analytics copilot. Explain holiday impact using ONLY the provided dataset summary.

Summary from enriched dataset:
- mapping: {mapping}
- holiday_count: {summary['holiday_count']}
- non_holiday_count: {summary['non_holiday_count']}
- holiday_avg_delay_hours: {summary['holiday_avg']}
- non_holiday_avg_delay_hours: {summary['non_holiday_avg']}
- ratio_holiday_vs_non_holiday: {summary['ratio']}
- top_holidays_by_delay: {summary.get('top_holidays', [])}

Output in markdown:
1) One short interpretation paragraph with the key comparison.
2) Up to 3 bullets with likely operational explanations grounded in this summary.
3) Up to 3 bullets with concrete follow-up checks.
Do not claim causality. Keep concise.
""".strip()

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=260,
        )
        text = resp.choices[0].message.content or ""
        return text.strip() or holiday_impact_fallback_markdown(summary)
    except Exception:
        return holiday_impact_fallback_markdown(summary)
