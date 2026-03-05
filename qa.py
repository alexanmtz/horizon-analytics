import re
import pandas as pd
import numpy as np
from metrics import compute_all_metrics, add_derived_columns


def answer_question(df, mapping, question: str) -> str:
    q_raw = (question or "")
    q = q_raw.lower()
    derived = add_derived_columns(df, mapping)
    scoped, scope_note = _apply_question_date_filter(derived, mapping, q_raw)

    # Priority path for holiday-specific intents so generic top-record logic
    # does not override the holiday-enriched table response.
    if "holiday" in q and "average" in q:
        if "is_bank_holiday" not in scoped.columns:
            return "Holiday enrichment is not connected yet. Ask me to suggest an external calendar datasource."

        if "delay_hours" not in scoped.columns:
            return "Delay metrics not available."

        holiday_avg = scoped.loc[
            scoped["is_bank_holiday"] & scoped["delay_hours"].notna() & (scoped["delay_hours"] > 0),
            "delay_hours"
        ].mean()

        non_avg = scoped.loc[
            (~scoped["is_bank_holiday"]) & scoped["delay_hours"].notna() & (scoped["delay_hours"] > 0),
            "delay_hours"
        ].mean()

        response = (
            "📊 Holiday vs Non-Holiday Delay Comparison\n\n"
            f"- Avg delay on holidays: **{holiday_avg:.2f}h**\n"
            f"- Avg delay on non-holidays: **{non_avg:.2f}h**\n"
            f"- Ratio: **{holiday_avg/non_avg:.2f}x slower on holidays**"
        )
        return _append_scope_note(response, scope_note)

    if "top" in q and "holiday" in q:
        if "is_bank_holiday" not in scoped.columns:
            return "Holiday enrichment is not connected yet. Ask me to suggest an external calendar datasource."

        if "holiday_name" not in scoped.columns:
            return "Holiday enrichment is connected, but holiday names are not available in this dataset."

        if "delay_hours" not in scoped.columns:
            return "Delay metrics not available."

        holiday_name_text = scoped["holiday_name"].fillna("").astype(str).str.strip()
        valid_holiday_name = holiday_name_text.ne("") & ~holiday_name_text.str.lower().isin(["nan", "none", "nat"])

        holiday_rows = scoped.loc[
            scoped["is_bank_holiday"]
            & scoped["delay_hours"].notna()
            & (scoped["delay_hours"] > 0)
            & valid_holiday_name
        ].copy()

        if holiday_rows.empty:
            return (
                "Holiday enrichment is connected, but there are no delayed payouts matched "
                "to named holidays in the current dataset/date range."
            )

        non_holiday_rows = scoped.loc[
            (~scoped["is_bank_holiday"])
            & scoped["delay_hours"].notna()
            & (scoped["delay_hours"] > 0)
        ].copy()

        if non_holiday_rows.empty:
            baseline_avg = float(holiday_rows["delay_hours"].mean())
            baseline_label = "all delayed rows"
        else:
            baseline_avg = float(non_holiday_rows["delay_hours"].mean())
            baseline_label = "non-holiday delayed rows"

        grouped = (
            holiday_rows.groupby("holiday_name")["delay_hours"]
            .agg(["mean", "count", "max"])
            .rename(columns={"mean": "avg_delay_hours", "count": "payout_count", "max": "max_delay_hours"})
            .reset_index()
        )

        if baseline_avg > 0:
            grouped["pct_above_avg"] = ((grouped["avg_delay_hours"] / baseline_avg) - 1.0) * 100.0
            grouped = grouped.loc[grouped["pct_above_avg"] >= 0].copy()
        else:
            grouped["pct_above_avg"] = 0.0

        grouped = grouped.sort_values(["pct_above_avg", "avg_delay_hours", "payout_count"], ascending=[False, False, False])

        cols = ["holiday_name", "payout_count", "avg_delay_hours", "max_delay_hours", "pct_above_avg"]
        table = format_table(grouped, cols, max_rows=max(10, int(len(grouped))))

        return _append_scope_note(
            (
                f"Holiday impact vs average delay baseline ({baseline_label}: **{baseline_avg:.2f}h**)\n\n"
                f"{table}\n\n"
                "`pct_above_avg` is how much higher each holiday's average delay is versus the baseline."
            ),
            scope_note,
        )

    if "holiday" in q and ("impact" in q or "explain" in q):
        if "is_bank_holiday" not in scoped.columns:
            return "Holiday enrichment is not connected yet. Ask me to suggest an external calendar datasource."

        if "delay_hours" not in scoped.columns:
            return "Delay metrics not available."

        holiday_delay = scoped.loc[
            scoped["is_bank_holiday"] & scoped["delay_hours"].notna() & (scoped["delay_hours"] > 0),
            "delay_hours"
        ]
        non_holiday_delay = scoped.loc[
            (~scoped["is_bank_holiday"]) & scoped["delay_hours"].notna() & (scoped["delay_hours"] > 0),
            "delay_hours"
        ]

        holiday_count = int(holiday_delay.shape[0])
        non_holiday_count = int(non_holiday_delay.shape[0])

        if holiday_count == 0:
            return (
                "Holiday enrichment is connected, but there are no delayed payout rows matched to holidays "
                "in the current dataset/date range."
            )

        holiday_avg = float(holiday_delay.mean()) if holiday_count > 0 else float("nan")
        non_avg = float(non_holiday_delay.mean()) if non_holiday_count > 0 else float("nan")

        ratio_text = "N/A"
        if pd.notna(non_avg) and non_avg > 0:
            ratio_text = f"{holiday_avg / non_avg:.2f}x"

        findings = [
            "🧠 Explanation",
            "",
            f"- Delayed holiday payouts analyzed: **{holiday_count}**",
            f"- Delayed non-holiday payouts analyzed: **{non_holiday_count}**",
            f"- Avg delay on holidays: **{holiday_avg:.2f}h**",
            f"- Avg delay on non-holidays: **{non_avg:.2f}h**" if pd.notna(non_avg) else "- Avg delay on non-holidays: **N/A**",
            f"- Relative holiday impact: **{ratio_text}**",
        ]

        if "holiday_name" in scoped.columns:
            top_holidays = (
                scoped.loc[scoped["is_bank_holiday"] & scoped["holiday_name"].astype(str).str.len().gt(0)]
                .groupby("holiday_name")["delay_hours"]
                .agg(["mean", "count"])
                .sort_values(["mean", "count"], ascending=[False, False])
                .head(3)
            )
            if not top_holidays.empty:
                findings.append("- Highest-delay holiday names in this data:")
                for name, row in top_holidays.iterrows():
                    findings.append(f"  - {name}: {row['mean']:.2f}h (n={int(row['count'])})")

        return _append_scope_note("\n".join(findings), scope_note)

    named_holiday_answer = _answer_named_holiday_question(scoped, q)
    if named_holiday_answer:
        return _append_scope_note(named_holiday_answer, scope_note)

    count_answer = _answer_count_question(scoped, mapping, q)
    if count_answer:
        return _append_scope_note(count_answer, scope_note)

    grouped_answer = _answer_grouped_aggregate_question(scoped, mapping, q)
    if grouped_answer:
        return _append_scope_note(grouped_answer, scope_note)

    summary_answer = _answer_metric_summary_question(scoped, q)
    if summary_answer:
        return _append_scope_note(summary_answer, scope_note)

    top_records_answer = _answer_top_records_question(scoped, mapping, q)
    if top_records_answer:
        return _append_scope_note(top_records_answer, scope_note)

    # 1️⃣ Basic metrics fallback
    if re.search(r"\b(average|mean|median|p90|delay|transfer)\b", q):
        return _append_scope_note(compute_all_metrics(scoped, mapping), scope_note)

    # 4️⃣ Why question (smart explanation)
    if "why" in q or "explain" in q:
        return _append_scope_note(_reason_discovery_summary(scoped, mapping), scope_note)

    # Fallback
    return _append_scope_note(compute_all_metrics(scoped, mapping), scope_note)


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _answer_named_holiday_question(derived: pd.DataFrame, q: str) -> str | None:
    if "holiday_name" not in derived.columns:
        return None

    holiday_series_all = derived["holiday_name"].fillna("").astype(str).str.strip()
    valid_mask = holiday_series_all.ne("") & ~holiday_series_all.str.lower().isin(["nan", "none", "nat"])
    holiday_series = holiday_series_all.loc[valid_mask]
    if holiday_series.empty:
        return None

    q_norm = _normalize_text_for_match(q)
    names = sorted(holiday_series.unique().tolist(), key=len, reverse=True)

    matched_name = None
    matched_name_norm = None
    q_padded = f" {q_norm} "
    for name in names:
        name_norm = _normalize_text_for_match(name)
        if not name_norm:
            continue
        if f" {name_norm} " in q_padded:
            matched_name = name
            matched_name_norm = name_norm
            break

    if not matched_name or not matched_name_norm:
        return None

    normalized_holidays = holiday_series_all.apply(_normalize_text_for_match)
    matched_rows = derived.loc[normalized_holidays == matched_name_norm].copy()

    is_count_intent = any(token in q for token in ["how many", "count", "total", "number of", "payout", "payouts"])
    is_delay_avg_intent = any(token in q for token in ["average", "avg", "mean"]) and any(
        token in q for token in ["delay", "late"]
    )

    if is_delay_avg_intent and "delay_hours" in matched_rows.columns:
        delay_values = pd.to_numeric(matched_rows["delay_hours"], errors="coerce").dropna()
        if delay_values.empty:
            return f"No valid delay values found for **{matched_name}** in the current dataset."
        return (
            f"Average delay on **{matched_name}**: **{delay_values.mean():.2f}h** "
            f"(rows: **{int(delay_values.shape[0])}**)."
        )

    if is_count_intent or " on " in q:
        return f"Payouts on **{matched_name}**: **{int(len(matched_rows))}**"

    return None


def _answer_grouped_aggregate_question(derived: pd.DataFrame, mapping: dict, q: str) -> str | None:
    if not _is_grouped_aggregate_intent(q):
        return None

    metric_col = _pick_metric_column(derived, q)
    if metric_col is None:
        return None

    dim_col = _pick_group_dimension(derived, mapping, q)
    if dim_col is None:
        return None

    filtered = derived.dropna(subset=[dim_col, metric_col]).copy()
    if filtered.empty:
        return f"I couldn't find enough data to compute `{metric_col}` by `{dim_col}`."

    ascending = _is_ascending_rank(q)

    grouped = (
        filtered.groupby(dim_col)[metric_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg", "count": "n"})
        .sort_values(["avg", "n"], ascending=[ascending, False])
        .reset_index()
    )

    if grouped.empty:
        return f"I couldn't compute grouped averages for `{metric_col}` by `{dim_col}`."

    top = grouped.head(10)
    result = format_table(top, [dim_col, "avg", "n"])

    lead = "Lowest" if ascending else "Top"
    return f"{lead} `{dim_col}` by average `{metric_col}`:\n\n{result}"


def _is_grouped_aggregate_intent(q: str) -> bool:
    has_metric = any(token in q for token in ["delay", "transfer", "processing", "late"])
    has_rank = any(token in q for token in ["highest", "lowest", "top", "best", "worst"])
    has_grouping = any(token in q for token in [" by ", " per ", "each", "group", "which source", "which provider", "which country"])
    has_plural_dim = any(token in q for token in ["sources", "providers", "countries", "currencies", "statuses", "banks"])
    has_dim_word = any(token in q for token in ["source", "provider", "country", "currency", "status", "bank", "weekday", "day of week"])
    return has_metric and (has_grouping or has_plural_dim or (has_rank and has_dim_word))


def _answer_metric_summary_question(derived: pd.DataFrame, q: str) -> str | None:
    metric_col = _pick_metric_column(derived, q)
    if metric_col is None:
        return None

    if not any(token in q for token in ["average", "avg", "mean", "median", "p90", "percentile", "max", "maximum", "min", "minimum", "overall"]):
        return None

    values = pd.to_numeric(derived[metric_col], errors="coerce").dropna()
    if values.empty:
        return f"I couldn't calculate `{metric_col}` because the data is missing or invalid."

    metric_label = metric_col.replace("_", " ")
    lines = [f"`{metric_label}` summary:", ""]

    if any(token in q for token in ["average", "avg", "mean", "overall"]):
        lines.append(f"- Mean: **{values.mean():.2f}**")
    if "median" in q:
        lines.append(f"- Median: **{values.median():.2f}**")
    if "p90" in q or "percentile" in q:
        lines.append(f"- P90: **{values.quantile(0.9):.2f}**")
    if any(token in q for token in ["max", "maximum", "highest"]):
        lines.append(f"- Max: **{values.max():.2f}**")
    if any(token in q for token in ["min", "minimum", "lowest"]):
        lines.append(f"- Min: **{values.min():.2f}**")

    if len(lines) == 2:
        lines.extend([
            f"- Mean: **{values.mean():.2f}**",
            f"- Median: **{values.median():.2f}**",
            f"- P90: **{values.quantile(0.9):.2f}**",
        ])

    lines.append(f"- Rows analyzed: **{int(values.shape[0])}**")
    return "\n".join(lines)


def _answer_top_records_question(derived: pd.DataFrame, mapping: dict, q: str) -> str | None:
    if not any(token in q for token in ["top", "largest", "biggest", "highest", "lowest", "smallest", "best"]):
        return None

    metric_col = _pick_metric_column(derived, q)
    if metric_col is None:
        return None

    values = derived.dropna(subset=[metric_col]).copy()
    if values.empty:
        return f"I couldn't find rows with `{metric_col}` values."

    ascending = _is_ascending_rank(q)
    n = _extract_limit(q, default=10, max_allowed=25)
    top = values.sort_values(metric_col, ascending=ascending).head(n)

    cols = [
        mapping.get("id"),
        "source",
        "provider",
        "country",
        metric_col,
        mapping.get("status"),
    ]
    cols = [c for c in cols if c and c in top.columns]
    if not cols:
        cols = [metric_col]

    lead = "Lowest" if ascending else "Top"
    return f"{lead} {n} rows by `{metric_col}`:\n\n{format_table(top, cols, max_rows=n)}"


def _answer_count_question(derived: pd.DataFrame, mapping: dict, q: str) -> str | None:
    if not any(token in q for token in ["how many", "count", "total", "number of", "breakdown", "by status", "per status"]):
        return None

    status_col = mapping.get("status") if mapping.get("status") in derived.columns else "status" if "status" in derived.columns else None

    if status_col:
        status_series = derived[status_col].astype(str).str.strip().str.lower()
        distinct_statuses = sorted([s for s in status_series.dropna().unique() if s and s != "nan"])
        for status in distinct_statuses:
            if re.search(rf"\b{re.escape(status)}\b", q):
                count = int((status_series == status).sum())
                return f"Rows with status `{status}`: **{count}**"

        if any(token in q for token in ["by status", "per status", "status breakdown", "statuses"]):
            breakdown = (
                derived.assign(_status=status_series)
                .groupby("_status")
                .size()
                .reset_index(name="count")
                .rename(columns={"_status": "status"})
                .sort_values("count", ascending=False)
            )
            if not breakdown.empty:
                return f"Status breakdown:\n\n{format_table(breakdown, ['status', 'count'], max_rows=20)}"

    if any(token in q for token in ["total", "all rows", "records", "payouts"]) and "by" not in q and "per" not in q:
        return f"Total rows in current dataset: **{len(derived)}**"

    return None


def _extract_limit(q: str, default: int = 10, max_allowed: int = 25) -> int:
    match = re.search(r"\b(top|lowest|bottom|best|highest|largest|smallest)\s+(\d{1,2})\b", q)
    if match:
        return max(1, min(int(match.group(2)), max_allowed))

    match = re.search(r"\b(\d{1,2})\b", q)
    if match and any(token in q for token in ["top", "highest", "largest", "biggest", "lowest", "smallest", "best", "bottom"]):
        return max(1, min(int(match.group(1)), max_allowed))

    return default


def _is_ascending_rank(q: str) -> bool:
    return any(token in q for token in ["lowest", "smallest", "best", "fastest", "least"])


def _append_scope_note(response: str, scope_note: str | None) -> str:
    if not scope_note:
        return response
    return f"{scope_note}\n\n{response}"


def _apply_question_date_filter(derived: pd.DataFrame, mapping: dict, question: str) -> tuple[pd.DataFrame, str | None]:
    start_ts, end_ts = _extract_date_range(question)
    if start_ts is None or end_ts is None:
        return derived, None

    date_col = _pick_date_column_for_filter(derived, mapping)
    if date_col is None:
        return derived, "Date filter detected, but no usable datetime column is available in this dataset."

    dt_series = pd.to_datetime(derived[date_col], errors="coerce", utc=True)
    mask = dt_series.between(start_ts, end_ts, inclusive="both")
    filtered = derived.loc[mask].copy()

    if filtered.empty:
        return filtered, (
            f"Applied date filter on `{date_col}` from `{start_ts.date()}` to `{end_ts.date()}` and found no matching rows."
        )

    note = (
        f"Applied date filter on `{date_col}` from `{start_ts.date()}` to `{end_ts.date()}` "
        f"(rows matched: **{len(filtered)}** of **{len(derived)}**)."
    )
    return filtered, note


def _extract_date_range(question: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not question:
        return None, None

    iso_date = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:Z)?)?"
    patterns = [
        rf"\bbetween\s+({iso_date})\s+and\s+({iso_date})\b",
        rf"\bfrom\s+({iso_date})\s+to\s+({iso_date})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue

        start = pd.to_datetime(match.group(1), errors="coerce", utc=True)
        end = pd.to_datetime(match.group(2), errors="coerce", utc=True)
        if pd.isna(start) or pd.isna(end):
            continue

        if start > end:
            start, end = end, start
        return start, end

    return None, None


def _pick_date_column_for_filter(derived: pd.DataFrame, mapping: dict) -> str | None:
    preferred = [
        mapping.get("arrival_at"),
        mapping.get("expected_arrival_at"),
        mapping.get("paid_at"),
        mapping.get("created_at"),
        "arrival_at",
        "arrived_at",
        "expected_arrival_at",
        "paid_at",
        "created_at",
    ]

    for col in preferred:
        if col and col in derived.columns:
            parsed = pd.to_datetime(derived[col], errors="coerce", utc=True)
            if parsed.notna().any():
                return col

    return None


def _pick_metric_column(derived: pd.DataFrame, q: str) -> str | None:
    has_delay_words = any(token in q for token in ["delay", "late", "arrival delay"])
    if has_delay_words and "delay_hours" in derived.columns:
        return "delay_hours"

    has_transfer_words = any(token in q for token in ["transfer", "processing", "processing time"])
    if has_transfer_words and "transfer_time_hours" in derived.columns:
        return "transfer_time_hours"

    if "delay_hours" in derived.columns and any(token in q for token in ["average", "mean", "highest", "top"]):
        return "delay_hours"

    return None


def _pick_group_dimension(derived: pd.DataFrame, mapping: dict, q: str) -> str | None:
    keyword_dimension_pairs = [
        (["source", "sources", "provider", "providers"], ["source", "provider", "bank", "bank_name"]),
        (["country", "countries"], ["country"]),
        (["currency", "currencies"], [mapping.get("currency"), "currency"]),
        (["status", "statuses", "state"], [mapping.get("status"), "status"]),
        (["weekday", "day of week", "day"], ["arrival_weekday"]),
    ]

    for keywords, candidates in keyword_dimension_pairs:
        if any(keyword in q for keyword in keywords):
            col = _first_existing_column(derived, candidates)
            if col is not None:
                return col

    if any(token in q for token in ["highest", "top", "which", "average", "mean"]):
        fallback_order = [
            "source",
            "provider",
            "bank",
            "bank_name",
            "country",
            mapping.get("currency"),
            "currency",
            mapping.get("status"),
            "status",
        ]
        return _first_existing_column(derived, fallback_order)

    return None


def _first_existing_column(df: pd.DataFrame, candidates: list[str | None]) -> str | None:
    for col in candidates:
        if col and col in df.columns:
            return col
    return None


def format_table(df: pd.DataFrame, cols: list[str], max_rows: int = 10) -> str:
    view = df[cols].head(max_rows).copy()

    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].round(2)

    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"

    rows = []
    for _, r in view.iterrows():
        row_values = [_format_cell(r[c]) for c in view.columns]
        rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join([header, sep] + rows)


def _format_cell(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.2f}"
    return str(v)


def _reason_discovery_summary(derived: pd.DataFrame, mapping: dict) -> str:
    if "delay_hours" not in derived.columns or derived["delay_hours"].notna().sum() == 0:
        return "I can't explain delay drivers yet because delay metrics are unavailable. Check timestamp mapping first."

    delay = derived["delay_hours"].dropna()
    baseline = delay.mean()
    findings = []

    candidate_dims = [
        "country",
        "bank",
        "bank_name",
        "provider",
        mapping.get("currency"),
        mapping.get("status"),
    ]
    candidate_dims = [c for c in candidate_dims if c and c in derived.columns]

    strongest_dim = None
    strongest_effect = 0.0

    for dim in candidate_dims:
        grp = derived.dropna(subset=[dim, "delay_hours"]).groupby(dim)["delay_hours"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 5]
        if grp.empty:
            continue

        span = float(grp["mean"].max() - grp["mean"].min())
        if span > strongest_effect:
            strongest_effect = span
            strongest_dim = dim

    if strongest_dim is not None:
        grp = derived.dropna(subset=[strongest_dim, "delay_hours"]).groupby(strongest_dim)["delay_hours"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 5].sort_values("mean", ascending=False)
        top_label = str(grp.index[0])
        top_mean = float(grp.iloc[0]["mean"])
        findings.append(f"Largest internal driver appears to be `{strongest_dim}` (top segment `{top_label}` avg `{top_mean:.2f}h`).")

    if "arrival_weekday" in derived.columns:
        w = derived.dropna(subset=["arrival_weekday", "delay_hours"]).groupby("arrival_weekday")["delay_hours"].mean()
        if not w.empty:
            peak_day = str(w.idxmax())
            peak_mean = float(w.max())
            findings.append(f"Delay also peaks on `{peak_day}` (`{peak_mean:.2f}h` average).")

    residual_hint = strongest_effect < max(2.0, baseline * 0.25)
    if residual_hint:
        findings.append("Internal columns only weakly explain delay variance; external calendar or operations context may help.")

    if not findings:
        return "I checked internal segments but found no strong delay drivers yet."

    return "🧠 Explanation\n\n- " + "\n- ".join(findings)