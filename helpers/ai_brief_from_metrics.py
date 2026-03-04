import pandas as pd
from clients.openai_client import client


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 5, max_cols: int = 8) -> str:
    preview = df.head(max_rows)
    cols = list(preview.columns)[:max_cols]
    preview = preview[cols] if cols else preview

    if preview.empty:
        return "| No data available |\n|---|"

    headers = [str(col) for col in preview.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    body_rows = []
    for _, row in preview.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            else:
                text = str(value).replace("\n", " ").replace("|", "\\|")
                values.append(text)
        body_rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header_row, separator_row, *body_rows])

async def ai_brief_from_metrics(
    df: pd.DataFrame,
    mapping: dict,
    metrics_md: str,
    timeout_seconds: float | None = 15.0,
) -> str:
    sample_cols = list(df.columns)[:40]
    sample_rows = df.head(5).to_dict(orient="records")
    data_preview_md = _to_markdown_table(df, max_rows=5, max_cols=8)

    prompt = f"""
You are an AI-first analytics assistant. Write a short "Initial Insight Brief" for this dataset.

You are given:
1) The computed metrics markdown (source of truth): 
{metrics_md}

2) Available columns (subset): {sample_cols}
3) Sample of the first 5 rows (enriched dataset): {sample_rows}
4) Column mapping: {mapping}
5) Data preview table to include in the response:
{data_preview_md}

Output (markdown):
- Start with "#### Data Preview" and include the provided markdown table exactly once.
- Then add 3–5 bullets on what stands out in this dataset (use numbers when available).
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

    resp = await client.responses.create(**req)
    return resp.output_text or ""