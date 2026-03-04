from clients.openai_client import client

async def ai_insight_brief(df, mapping, metrics_text: str):
    # small sample only
    sample = df.head(8).to_dict(orient="records")

    prompt = f"""
You are an analytics copilot. Create a short "Initial Insight Brief" for the user.

Context:
- mapping: {mapping}
- precomputed metrics (may be partial): {metrics_text}
- sample rows: {sample}

Requirements:
- 3-5 bullets of key observations
- 2 bullets: what could explain delays (calendar/cutoffs/provider processing)
- 3 suggested next questions the user can ask
Keep it concise.
""".strip()

    resp = await client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": prompt}],
    )
    return resp.output_text or ""