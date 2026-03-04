from clients.openai_client import client

def _parse_bullets(text: str) -> list[str]:
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    qs = []
    for l in lines:
        l = l.lstrip("-• ").strip()
        if l:
            qs.append(l)
    # keep unique, keep first 3
    out = []
    seen = set()
    for q in qs:
        k = q.lower()
        if k not in seen:
            out.append(q)
            seen.add(k)
    return out[:3]

async def suggest_followups(question: str, result_summary: str, profile: dict) -> list[str]:
    prompt = f"""
You are an analytics copilot.

User asked:
{question}

Result summary:
{result_summary}

Dataset constraints:
- columns: {profile.get("columns")}
- mapping: {profile.get("mapping")}
- has_delay: {profile.get("has_delay")}
- has_transfer_time: {profile.get("has_transfer_time")}

Task:
Suggest exactly 3 short follow-up questions the user could ask next
that can be answered using ONLY the available columns/derived fields.

Rules:
- each under 12 words
- actionable (group by, trend, top N, correlation)
- avoid repeating the same idea
Return only a bullet list.
""".strip()

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )

    text = resp.choices[0].message.content
    return _parse_bullets(text)