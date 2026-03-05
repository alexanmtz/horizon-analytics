import pandas as pd

from clients.openai_client import call_openai_with_retries, client

async def ai_brief_from_metrics(
    df: pd.DataFrame,
    mapping: dict,
    metrics_md: str,
    timeout_seconds: float | None = 15.0,
) -> str:
    sample_cols = list(df.columns)[:40]
    sample_rows = df.head(5).to_dict(orient="records")

    prompt = f"""
You are an AI-first analytics assistant. Write a short "Initial Insight Brief" for this dataset.

You are given:
1) The computed metrics markdown (source of truth): 
{metrics_md}

2) Available columns (subset): {sample_cols}
3) Sample of the first 5 rows (enriched dataset): {sample_rows}
4) Column mapping: {mapping}

Output (markdown):
- Add 3–5 bullets on what stands out in this dataset (use numbers when available).
- Add 2 bullets with plausible drivers/patterns (generic, no domain assumptions).
- Add 3 bullets with actionable next questions the user should ask.
Keep it concise and generic. Do not assume delay-related columns exist.
""".strip()

    req = {
        "model": "gpt-4.1-mini",
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 350,
    }
    if timeout_seconds is not None:
        req["timeout"] = timeout_seconds

    resp = await call_openai_with_retries(lambda: client.responses.create(**req))
    if resp:
        text = (resp.output_text or "").strip()
        if text:
            return text

    return "\n".join([
        "- Initial metrics were generated successfully.",
        "- AI summary is temporarily unavailable due to model service instability.",
        "- Ask a specific question to continue analysis from computed metrics.",
    ])