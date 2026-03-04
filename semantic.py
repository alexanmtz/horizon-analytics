import re
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

PATTERNS = {
    "id": [r"\bid\b", r"payout_id", r"transfer_id", r"transaction_id"],
    "status": [r"status", r"state"],
    "amount": [r"amount", r"net", r"gross", r"total", r"value"],
    "currency": [r"currency", r"ccy"],
    "created_at": [r"created", r"created_at", r"creation"],
    "paid_at": [r"paid", r"paid_at", r"initiated", r"sent_at"],
    "expected_arrival_at": [r"expected.*arrival", r"eta", r"expected_at"],
    "arrival_at": [r"arrival", r"arrived", r"settled", r"landing", r"available_at"],
}


def suggest_mapping(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    mapping = {k: None for k in DEFAULT_KEYS}

    # 1) name-based regex
    for key, pats in PATTERNS.items():
        for col in cols:
            for pat in pats:
                if re.search(pat, col, flags=re.IGNORECASE):
                    mapping[key] = col
                    break
            if mapping[key]:
                break

    # 2) heuristic: detect datetime/parseable columns
    dt_candidates = guess_datetime_cols(df)
    # If arrival/paid/etc was not found, fallback to best candidates
    if mapping["created_at"] is None and dt_candidates:
        mapping["created_at"] = dt_candidates[0]
    if mapping["paid_at"] is None and len(dt_candidates) > 1:
        mapping["paid_at"] = dt_candidates[1]
    if mapping["arrival_at"] is None and len(dt_candidates) > 2:
        mapping["arrival_at"] = dt_candidates[2]

    return mapping


def guess_datetime_cols(df: pd.DataFrame) -> list[str]:
    candidates = []
    for c in df.columns:
        if "date" in c or c.endswith("_at") or "time" in c:
            candidates.append(c)

    # try parsing candidates
    ok = []
    for c in candidates:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().mean() > 0.6:  # 60% parseable
                ok.append(c)
        except Exception:
            continue

    # sort with preference: arrival/paid/created
    priority = {"arrival": 0, "paid": 1, "created": 2}
    def score(col: str) -> tuple[int, str]:
        for k, v in priority.items():
            if k in col:
                return (v, col)
        return (9, col)

    return sorted(ok, key=score)


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