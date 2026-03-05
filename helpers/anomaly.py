import pandas as pd
from clients.openai_client import call_openai_with_retries, client

def detect_biggest_anomaly(df: pd.DataFrame, mapping: dict) -> dict | None:
    """
    Picks 1 "most unusual" row using delay_hours or transfer_time_hours.
    Returns a dict with the row + a short numeric summary.
    """
    candidates = []
    for col in ["delay_hours", "transfer_time_hours"]:
        if col in df.columns and df[col].notna().any():
            s = df[col].dropna()
            mean = float(s.mean())
            std = float(s.std(ddof=0))  # population std to avoid tiny sample oddities
            if std > 0:
                z = ((df[col] - mean).abs() / std)
                idx = z.idxmax()
                candidates.append((float(z.loc[idx]), col, idx, mean, std))

    if not candidates:
        return None

    z, col, idx, mean, std = sorted(candidates, reverse=True)[0]
    row = df.loc[idx].to_dict()

    # Choose a few useful fields to show (if present)
    cols_show = [
        mapping.get("id"),
        col,
        mapping.get("status"),
        mapping.get("currency"),
        mapping.get("amount"),
        mapping.get("arrival_at"),
        mapping.get("expected_arrival_at"),
        mapping.get("paid_at"),
    ]
    cols_show = [c for c in cols_show if c and c in df.columns]

    row_compact = {k: row.get(k) for k in cols_show}
    return {
        "metric": col,
        "z_score": z,
        "mean": mean,
        "std": std,
        "row": row_compact,
    }


async def ai_explain_anomaly(anomaly: dict, mapping: dict, has_holidays: bool = False) -> str:
    prompt = f"""
You are an AI analytics copilot. Explain the single most unusual record detected.

Context:
- mapping: {mapping}
- holiday_calendar_connected: {has_holidays}

Anomaly summary:
- metric: {anomaly["metric"]}
- z_score: {anomaly["z_score"]:.2f}
- mean: {anomaly["mean"]:.2f}
- std: {anomaly["std"]:.2f}
- row fields: {anomaly["row"]}

Write:
1) What makes it unusual (use the numbers)
2) 3 plausible causes (calendar/cutoff/provider/bank batching/data quality)
3) 3 targeted checks we can run next (group by weekday/provider/status/etc.)
Keep it concise in markdown.
""".strip()

    resp = await call_openai_with_retries(
        lambda: client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
    )
    if resp:
        text = resp.choices[0].message.content or ""
        if text.strip():
            return text.strip()

    return "\n".join([
        "- This record is unusual because its value is far from the dataset average.",
        f"- Metric: **{anomaly['metric']}**, z-score: **{anomaly['z_score']:.2f}**.",
        "- Likely causes: cutoff timing, provider batching, or data quality issues.",
        "- Next checks: group by weekday/provider/status and inspect neighboring timestamps.",
    ])