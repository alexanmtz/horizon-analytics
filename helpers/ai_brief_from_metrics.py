import pandas as pd
from clients.openai_client import client

async def ai_brief_from_metrics(
    df: pd.DataFrame,
    mapping: dict,
    metrics_md: str,
    timeout_seconds: float | None = 15.0,
) -> str:
    # Small payload: a couple of rows from the top delays + columns list
    sample_cols = list(df.columns)[:40]
    top_delays = []
    if "delay_hours" in df.columns and df["delay_hours"].notna().any():
        top_delays = df.sort_values("delay_hours", ascending=False).head(5).to_dict(orient="records")

    prompt = f"""
You are an AI-first analytics assistant. Write a short "Initial Insight Brief" for this dataset.

You are given:
1) The computed metrics markdown (source of truth): 
{metrics_md}

2) Available columns (subset): {sample_cols}
3) Sample of the 5 largest-delay rows: {top_delays}
4) Column mapping: {mapping}

Output (markdown):
- 3–5 bullets: what stands out (use numbers if present)
- 2 bullets: likely explanations for delays (calendar/cutoffs/provider processing/bank batching)
- 3 bullets: suggested next questions the user should ask (actionable)
Keep it concise. Do NOT repeat the full tables.
""".strip()

    req = {
        "model": "gpt-4.1-mini",
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 350,
    }
    if timeout_seconds is not None:
        req["timeout"] = timeout_seconds

    resp = await client.responses.create(**req)
    return resp.output_text or ""