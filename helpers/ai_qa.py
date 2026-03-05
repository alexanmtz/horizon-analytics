import json
import re
import pandas as pd

from clients.openai_client import call_openai_with_retries, client
from metrics import add_derived_columns


def should_use_llm_for_question(question: str, deterministic_answer: str) -> bool:
    q = (question or "").lower()
    generic = (deterministic_answer or "").strip().startswith("### 📌 Key metrics")

    holiday_intent = any(
        token in q
        for token in [
            "holiday",
            "christmas",
            "xmas",
            "easter",
            "good friday",
            "pentecost",
            "new year",
            "bank holiday",
        ]
    )

    complex_intent = any(
        token in q
        for token in [
            "correl",
            "relationship",
            "related",
            "impact",
            "driver",
            "drivers",
            "on sunday",
            "on mond",
            "on tues",
            "on wednes",
            "on thurs",
            "on fri",
            "on satur",
            "weekday",
            "weekend",
        ]
    )

    named_event_count_intent = bool(
        re.search(r"\b(how many|count|number of)\b.*\bon\b\s+[a-z]", q)
    )

    return generic or complex_intent or holiday_intent or named_event_count_intent


async def ai_answer_question(
    df: pd.DataFrame,
    mapping: dict,
    question: str,
    deterministic_answer: str,
    timeout_seconds: float = 20.0,
) -> str:
    derived = add_derived_columns(df, mapping)
    context = _build_context(derived, mapping, question)

    prompt = f"""
You are a data analyst assistant. Answer the user's question using only the provided dataset context.

User question:
{question}

Deterministic baseline answer (may be generic):
{deterministic_answer}

Dataset context (JSON):
{json.dumps(context, ensure_ascii=False)}

Rules:
- Be concrete and data-grounded.
- If asked about correlation, mention direction and magnitude from provided stats.
- If the question contains a segment (e.g., Sundays), prioritize that segment in the answer.
- If data is insufficient, explicitly say what is missing.
- Keep response concise (4-8 bullets max).
- Do not invent columns or values not present in context.
""".strip()

    req = {
        "model": "gpt-4.1-mini",
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 320,
        "timeout": timeout_seconds,
    }
    resp = await call_openai_with_retries(lambda: client.responses.create(**req))
    if resp:
        text = (resp.output_text or "").strip()
        if text:
            return text

    return deterministic_answer


def _build_context(derived: pd.DataFrame, mapping: dict, question: str) -> dict:
    q = (question or "").lower()
    context = {
        "rows": int(len(derived)),
        "columns": list(derived.columns)[:80],
        "mapping": mapping,
        "metrics": {},
        "segment_insights": {},
        "sample_rows": [],
    }

    for col in ["delay_hours", "transfer_time_hours"]:
        if col in derived.columns:
            s = pd.to_numeric(derived[col], errors="coerce").dropna()
            if not s.empty:
                context["metrics"][col] = {
                    "count": int(s.shape[0]),
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "p90": round(float(s.quantile(0.9)), 4),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                }

    amount_col = mapping.get("amount") if mapping.get("amount") in derived.columns else "amount" if "amount" in derived.columns else None
    if amount_col:
        amount = pd.to_numeric(derived[amount_col], errors="coerce")
        delay = pd.to_numeric(derived["delay_hours"], errors="coerce") if "delay_hours" in derived.columns else None

        if delay is not None:
            pair = pd.DataFrame({"amount": amount, "delay_hours": delay}).dropna()
            if pair.shape[0] >= 3:
                context["segment_insights"]["corr_amount_delay_hours"] = round(float(pair["amount"].corr(pair["delay_hours"])), 4)

    if "arrival_weekday" in derived.columns and "delay_hours" in derived.columns:
        weekday_grp = (
            derived.dropna(subset=["arrival_weekday", "delay_hours"])
            .groupby("arrival_weekday")["delay_hours"]
            .agg(["mean", "count"])
        )
        if not weekday_grp.empty:
            weekday_table = weekday_grp.sort_values("mean", ascending=False).reset_index()
            context["segment_insights"]["delay_by_weekday"] = weekday_table.to_dict(orient="records")

            sunday = weekday_table.loc[weekday_table["arrival_weekday"].astype(str).str.lower() == "sunday"]
            if not sunday.empty:
                context["segment_insights"]["sunday_delay"] = {
                    "mean": round(float(sunday.iloc[0]["mean"]), 4),
                    "count": int(sunday.iloc[0]["count"]),
                }

            if amount_col and re.search(r"\bsunday\b", q, flags=re.IGNORECASE):
                sunday_rows = derived.loc[
                    derived["arrival_weekday"].astype(str).str.lower() == "sunday",
                    [amount_col, "delay_hours"],
                ].copy()
                sunday_rows[amount_col] = pd.to_numeric(sunday_rows[amount_col], errors="coerce")
                sunday_rows["delay_hours"] = pd.to_numeric(sunday_rows["delay_hours"], errors="coerce")
                sunday_rows = sunday_rows.dropna()
                if sunday_rows.shape[0] >= 3:
                    context["segment_insights"]["corr_amount_delay_sunday"] = round(
                        float(sunday_rows[amount_col].corr(sunday_rows["delay_hours"])),
                        4,
                    )
                    context["segment_insights"]["sunday_corr_n"] = int(sunday_rows.shape[0])

    if "holiday_name" in derived.columns:
        holiday_series = derived["holiday_name"].fillna("").astype(str).str.strip()
        valid_mask = holiday_series.ne("") & ~holiday_series.str.lower().isin(["nan", "none", "nat"])

        if valid_mask.any():
            holiday_counts = (
                holiday_series.loc[valid_mask]
                .value_counts()
                .head(30)
                .rename_axis("holiday_name")
                .reset_index(name="count")
            )
            context["segment_insights"]["holiday_name_counts"] = holiday_counts.to_dict(orient="records")

            holiday_name_norm = holiday_series.apply(
                lambda v: re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()
            )
            q_norm = re.sub(r"[^a-z0-9]+", " ", q).strip()
            matched_holiday_norms = [
                name
                for name in holiday_name_norm.loc[valid_mask].dropna().unique().tolist()
                if name and f" {name} " in f" {q_norm} "
            ]

            if matched_holiday_norms:
                matched_mask = holiday_name_norm.isin(matched_holiday_norms)
                matched_count = int(matched_mask.sum())
                context["segment_insights"]["question_matched_holiday_rows"] = matched_count

                matched_names = sorted(holiday_series.loc[matched_mask].dropna().astype(str).unique().tolist())
                context["segment_insights"]["question_matched_holidays"] = matched_names[:10]

                if "delay_hours" in derived.columns:
                    matched_delay = pd.to_numeric(derived.loc[matched_mask, "delay_hours"], errors="coerce").dropna()
                    if not matched_delay.empty:
                        context["segment_insights"]["question_matched_holiday_delay"] = {
                            "mean": round(float(matched_delay.mean()), 4),
                            "median": round(float(matched_delay.median()), 4),
                            "count": int(matched_delay.shape[0]),
                        }

    preview_cols = [c for c in [mapping.get("id"), amount_col, "delay_hours", "arrival_weekday", mapping.get("status")] if c and c in derived.columns]
    if "holiday_name" in derived.columns:
        preview_cols.append("holiday_name")
    if "is_bank_holiday" in derived.columns:
        preview_cols.append("is_bank_holiday")
    preview_cols = list(dict.fromkeys(preview_cols))
    if preview_cols:
        context["sample_rows"] = derived[preview_cols].head(20).to_dict(orient="records")

    return context