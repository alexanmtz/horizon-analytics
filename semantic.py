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
    return mapping


def guess_datetime_cols(df: pd.DataFrame) -> list[str]:
    ok = []
    for c in df.columns:
        try:
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

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    datetime_ratio = float(dt.notna().mean())

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
    temporal_keys = ["created_at", "paid_at", "expected_arrival_at", "arrival_at"]
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
        v = v.strip()
        updated[k] = v

    return updated