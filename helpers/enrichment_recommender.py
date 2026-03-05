import json
from typing import Any
import pandas as pd

from clients.openai_client import call_openai_with_retries, client


CATALOG = [
    {
        "id": "openholidays",
        "name": "OpenHolidays PublicHolidays",
        "base_url": "https://openholidaysapi.org/PublicHolidays",
        "supports": ["calendar", "country", "date"],
        "requires": ["country", "expected_arrival_at"],
        "adds": ["is_bank_holiday", "holiday_name"],
    }
]


def _compact_profile(df: pd.DataFrame, mapping: dict) -> dict[str, Any]:
    cols = list(df.columns)
    return {
        "rows": int(len(df)),
        "columns": cols[:80],
        "mapping": mapping,
        "has_country": "country" in cols,
        "has_delay": "delay_hours" in cols,
        "country_cardinality": int(df["country"].nunique()) if "country" in cols else 0,
    }


async def recommend_enrichment(df: pd.DataFrame, mapping: dict, question: str, answer: str) -> dict | None:
    profile = _compact_profile(df, mapping)

    prompt = f"""
You are an analytics enrichment planner.

Inputs:
- user question: {question}
- current answer: {answer}
- dataset profile: {profile}
- datasource catalog: {CATALOG}

Task:
Return JSON only with this schema:
{{
  "recommend": true|false,
  "source_id": "openholidays"|null,
  "reason": "short reason",
  "confidence": 0.0-1.0
}}

Rules:
- Recommend only if current data likely cannot fully explain the question.
- Prefer internal analysis first; avoid external API if not needed.
- If the question is about delays, timing patterns, weekdays, settlement windows, or calendar effects and country/date context exists, openholidays is valid.
- If requirements are missing (country/date), set recommend=false.
""".strip()

    parsed = None
    resp = await call_openai_with_retries(
        lambda: client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=180,
        )
    )
    if resp:
        try:
            raw = (resp.output_text or "").strip()
            parsed = json.loads(raw)
        except Exception:
            parsed = None

    if parsed is None:
        parsed = {
            "recommend": False,
            "source_id": None,
            "reason": "No enrichment recommendation available.",
            "confidence": 0.0,
        }

    if not parsed.get("recommend"):
        return None

    source_id = parsed.get("source_id")
    if source_id != "openholidays":
        return None

    if "country" not in df.columns:
        return None

    exp_col = mapping.get("expected_arrival_at")
    if not exp_col or exp_col not in df.columns:
        return None

    return {
        "source_id": source_id,
        "reason": parsed.get("reason") or "External calendar may explain residual delay patterns.",
        "confidence": float(parsed.get("confidence", 0.5) or 0.5),
    }
