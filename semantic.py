from difflib import SequenceMatcher
import pandas as pd


DEFAULT_KEYS = [
    "id",
    "status",
    "amount",
    "currency",
    "created_at",
    "paid_at",
    "expected_arrival_at",
    "arrival_at",
]

CORE_TEMPORAL_KEYS = ["arrival_at", "expected_arrival_at", "paid_at", "created_at"]

def suggest_mapping(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    mapping = {k: None for k in DEFAULT_KEYS}
    profiles = {col: _column_profile(df, col) for col in cols}

    score_rows: list[tuple[str, str, float]] = []
    for key in DEFAULT_KEYS:
        for col in cols:
            name_score = _name_similarity_score(key, col)
            behavior_score = _behavior_score(key, profiles[col])
            total = (0.55 * name_score) + (0.45 * behavior_score)
            score_rows.append((key, col, total))

    # Greedy assignment with conflict resolution by highest score.
    score_rows.sort(key=lambda row: row[2], reverse=True)
    used_cols = set()
    for key, col, score in score_rows:
        if mapping[key] is not None:
            continue
        if col in used_cols:
            continue
        if score < 0.33:
            continue
        mapping[key] = col
        used_cols.add(col)

    _fill_temporal_fallbacks(df, mapping)
    _attach_dynamic_column_keys(df, mapping)
    return mapping


def _attach_dynamic_column_keys(df: pd.DataFrame, mapping: dict) -> None:
    """
    Attach dynamic keys for unmapped columns using normalized column names.
    Example: "Customer Country" -> mapping["customer_country"] = "Customer Country"
    """
    if df is None or df.empty:
        return

    used_cols = {v for v in mapping.values() if isinstance(v, str) and v}

    for col in df.columns:
        col_name = str(col)
        if col_name in used_cols:
            continue

        dynamic_key = _normalize_column_ref(col_name)
        if not dynamic_key:
            continue

        if dynamic_key in DEFAULT_KEYS:
            continue

        current = mapping.get(dynamic_key)
        if current and current != col_name:
            continue

        mapping[dynamic_key] = col_name


def guess_datetime_cols(df: pd.DataFrame) -> list[str]:
    ok = []
    for c in df.columns:
        try:
            if not _is_temporal_candidate(df[c], c):
                continue
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().mean() > 0.6:  # 60% parseable
                ok.append(c)
        except Exception:
            continue
    return ok


def _column_profile(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n = max(len(s), 1)

    notna = s.notna().sum()
    unique = s.dropna().nunique()

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())

    if _is_temporal_candidate(s, col):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        datetime_ratio = float(dt.notna().mean())
    else:
        datetime_ratio = 0.0

    str_s = s.dropna().astype(str)
    avg_len = float(str_s.str.len().mean()) if len(str_s) > 0 else 0.0
    alpha_ratio = float(str_s.str.fullmatch(r"[A-Za-z]+", na=False).mean()) if len(str_s) > 0 else 0.0

    return {
        "notna_ratio": float(notna / n),
        "unique_ratio": float(unique / max(notna, 1)),
        "numeric_ratio": numeric_ratio,
        "datetime_ratio": datetime_ratio,
        "avg_len": avg_len,
        "alpha_ratio": alpha_ratio,
    }


def _is_temporal_candidate(series: pd.Series, col: str) -> bool:
    name = str(col).lower()
    temporal_name_hints = ["date", "time", "timestamp", "datetime", "_at", "arrival", "paid", "created", "expected"]

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if pd.api.types.is_numeric_dtype(series):
        return any(hint in name for hint in temporal_name_hints)

    # For object/string-like columns, require either temporal naming hints
    # or timestamp-like characters in a meaningful portion of the values.
    if any(hint in name for hint in temporal_name_hints):
        return True

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False

    sample = non_null.head(200)
    ts_like_ratio = sample.str.contains(r"[-/:T ]", regex=True, na=False).mean()
    return float(ts_like_ratio) > 0.4


def _name_similarity_score(key: str, col: str) -> float:
    key_tokens = [t for t in key.lower().split("_") if t and t not in {"at"}]
    col_tokens = [t for t in str(col).lower().split("_") if t and t not in {"at"}]

    if not key_tokens or not col_tokens:
        return 0.0

    overlap = len(set(key_tokens) & set(col_tokens))
    token_score = overlap / max(len(set(key_tokens) | set(col_tokens)), 1)
    seq_score = SequenceMatcher(None, key.lower(), str(col).lower()).ratio()

    return float((0.6 * token_score) + (0.4 * seq_score))


def _behavior_score(key: str, p: dict) -> float:
    if key in {"created_at", "paid_at", "expected_arrival_at", "arrival_at"}:
        return p["datetime_ratio"]

    if key == "amount":
        dense_numeric = p["numeric_ratio"]
        variability = min(1.0, p["unique_ratio"] * 1.5)
        return float((0.7 * dense_numeric) + (0.3 * variability))

    if key == "currency":
        short_codes = max(0.0, 1.0 - abs(p["avg_len"] - 3.0) / 5.0)
        low_cardinality = max(0.0, 1.0 - p["unique_ratio"])
        return float((0.45 * p["alpha_ratio"]) + (0.35 * short_codes) + (0.20 * low_cardinality))

    if key == "status":
        low_cardinality = max(0.0, 1.0 - p["unique_ratio"])
        non_numeric = 1.0 - p["numeric_ratio"]
        return float((0.65 * low_cardinality) + (0.35 * non_numeric))

    if key == "id":
        non_datetime = 1.0 - p["datetime_ratio"]
        high_uniqueness = p["unique_ratio"]
        return float((0.55 * high_uniqueness) + (0.45 * non_datetime))

    return 0.0


def _fill_temporal_fallbacks(df: pd.DataFrame, mapping: dict) -> None:
    temporal_keys = CORE_TEMPORAL_KEYS
    unresolved = [k for k in temporal_keys if not mapping.get(k)]
    if not unresolved:
        return

    dt_cols = guess_datetime_cols(df)
    dt_cols = [c for c in dt_cols if c not in set(mapping.values())]
    if not dt_cols:
        return

    def dt_median(col: str):
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        return parsed.dropna().median()

    ranked = []
    for c in dt_cols:
        med = dt_median(c)
        if med is not pd.NaT:
            ranked.append((c, med))

    if not ranked:
        return

    ranked.sort(key=lambda row: row[1])
    ordered_cols = [c for c, _ in ranked]

    for key, col in zip(unresolved, ordered_cols):
        mapping[key] = col


def apply_mapping_override(current: dict, command: str) -> dict:
    """
    command: "map arrival_at=arrivalDate paid_at=paidDate"
    """
    parts = command.strip().split()[1:]  # remove 'map'
    updated = dict(current)

    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip().strip("`\"'")
        updated[k] = v

    return updated


def validate_temporal_mapping(df: pd.DataFrame, mapping: dict, min_parse_ratio: float = 0.6) -> tuple[dict, list[str]]:
    """
    Validates temporal keys and auto-repairs invalid mappings using best datetime candidates.
    Returns (updated_mapping, warnings).
    """
    updated = dict(mapping or {})
    warnings: list[str] = []
    missing_keys: dict[str, str] = {}

    temporal_keys = CORE_TEMPORAL_KEYS
    for k in temporal_keys:
        updated.setdefault(k, None)

    # Validate assigned columns first.
    used_by_temporal = {
        col for key, col in updated.items()
        if key in temporal_keys and isinstance(col, str) and col in df.columns
    }

    for key in temporal_keys:
        col = updated.get(key)
        if not col:
            continue
        original_col = col
        resolved_col = _resolve_column_name(df, col)
        if not resolved_col:
            missing_keys[key] = str(col)
            used_by_temporal.discard(col)
            updated[key] = None
            continue

        if resolved_col != col:
            warnings.append(f"Normalized {key} mapping: {col} -> {resolved_col}.")
            updated[key] = resolved_col
        col = resolved_col
        used_by_temporal.discard(original_col)
        used_by_temporal.add(col)

        ratio = _datetime_parse_ratio(df[col])
        if not _is_temporal_candidate(df[col], col):
            ratio = 0.0
        if ratio < min_parse_ratio:
            warnings.append(
                f"{key} -> {col}: only {ratio:.0%} parseable datetimes (min {min_parse_ratio:.0%})."
            )
            used_by_temporal.discard(col)
            updated[key] = None

    # Repair unresolved temporal keys with best available datetime candidates.
    for key in temporal_keys:
        if updated.get(key):
            continue

        best_col = _pick_best_temporal_replacement(
            df=df,
            key=key,
            excluded_cols=used_by_temporal,
            min_parse_ratio=min_parse_ratio,
        )
        if best_col:
            updated[key] = best_col
            used_by_temporal.add(best_col)
            warnings.append(f"Auto-mapped {key} -> {best_col}.")
            if key in missing_keys:
                missing_keys.pop(key, None)

    for key, col in missing_keys.items():
        warnings.append(f"{key} -> {col}: column not found.")

    return updated, warnings


def _resolve_column_name(df: pd.DataFrame, raw_col: str | None) -> str | None:
    if raw_col is None:
        return None

    candidate = str(raw_col).strip().strip("`\"'")
    if not candidate:
        return None

    cols = [str(c) for c in df.columns]

    if candidate in cols:
        return candidate

    by_stripped = {c.strip(): c for c in cols}
    if candidate in by_stripped:
        return by_stripped[candidate]

    lower_map = {c.strip().lower(): c for c in cols}
    lower_candidate = candidate.lower()
    if lower_candidate in lower_map:
        return lower_map[lower_candidate]

    norm_candidate = _normalize_column_ref(candidate)
    norm_to_col: dict[str, str | None] = {}
    for col in cols:
        norm_col = _normalize_column_ref(col)
        if norm_col not in norm_to_col:
            norm_to_col[norm_col] = col
        else:
            norm_to_col[norm_col] = None

    resolved = norm_to_col.get(norm_candidate)
    if resolved:
        return resolved

    return None


def _normalize_column_ref(name: str) -> str:
    s = str(name).strip()
    s = s.replace(" ", "_").replace("-", "_").replace("/", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.lower()


def _datetime_parse_ratio(series: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(series):
        return float(series.notna().mean())

    if pd.api.types.is_numeric_dtype(series):
        return 0.0

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return float(parsed.notna().mean())


def _pick_best_temporal_replacement(
    df: pd.DataFrame,
    key: str,
    excluded_cols: set[str],
    min_parse_ratio: float,
) -> str | None:
    best_col = None
    best_score = -1.0

    for col in df.columns:
        if col in excluded_cols:
            continue

        series = df[col]
        if not _is_temporal_candidate(series, col):
            continue

        parse_ratio = _datetime_parse_ratio(series)
        if parse_ratio < min_parse_ratio:
            continue

        name_score = _name_similarity_score(key, col)
        affinity = _temporal_name_affinity(key, col)
        score = (0.55 * name_score) + (0.20 * parse_ratio) + (0.25 * affinity)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _temporal_name_affinity(key: str, col: str) -> float:
    name = str(col).lower()

    def has_any(tokens: list[str]) -> bool:
        return any(t in name for t in tokens)

    if key == "arrival_at":
        if has_any(["arriv", "arrival"]):
            if has_any(["expect", "eta", "estimated"]):
                return 0.2
            return 1.0
        return 0.0

    if key == "expected_arrival_at":
        if has_any(["expect", "eta", "estimated"]):
            return 1.0
        if has_any(["arriv", "arrival"]):
            return 0.35
        return 0.0

    if key == "paid_at":
        if has_any(["paid", "pay", "settl", "release"]):
            return 1.0
        return 0.0

    if key == "created_at":
        if has_any(["creat", "submitted", "initiated", "request"]):
            return 1.0
        return 0.0

    return 0.0