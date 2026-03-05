from clients.openai_client import call_openai_with_retries, client

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


def _fallback_followups(profile: dict) -> list[str]:
    has_delay = bool(profile.get("has_delay"))
    has_transfer_time = bool(profile.get("has_transfer_time"))
    columns = set(profile.get("columns") or [])

    ideas: list[str] = []
    if has_delay:
        ideas.append("Which segments show the highest average delay?")
    if has_transfer_time:
        ideas.append("How has transfer time changed over time?")
    if "status" in columns:
        ideas.append("What are the top statuses by total amount?")
    ideas.append("Which top 5 entities drive most of the volume?")
    ideas.append("What trend appears by week in this dataset?")

    unique: list[str] = []
    seen: set[str] = set()
    for idea in ideas:
        key = idea.lower()
        if key not in seen:
            unique.append(idea)
            seen.add(key)
        if len(unique) == 3:
            break

    return unique or [
        "Which groups contribute most to the totals?",
        "What trend is visible over time?",
        "Which outliers should we investigate next?",
    ]

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

    resp = await call_openai_with_retries(
        lambda: client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
    )
    if resp:
        text = (resp.choices[0].message.content or "")
        parsed = _parse_bullets(text)
        if parsed:
            return parsed

    return _fallback_followups(profile)