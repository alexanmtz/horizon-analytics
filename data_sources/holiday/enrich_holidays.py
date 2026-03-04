import pandas as pd


def enrich_with_holidays(payouts: pd.DataFrame, holidays: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Adds:
      expected_date (YYYY-MM-DD)
      is_bank_holiday (bool)
      holiday_name (str)

    Join keys:
      payouts.country + expected_date  <->  holidays.country + holidays.date
    """
    d = payouts.copy()

    exp_col = mapping.get("expected_arrival_at")
    if not exp_col or exp_col not in d.columns:
        d["expected_date"] = ""
        d["is_bank_holiday"] = False
        d["holiday_name"] = ""
        return d

    if "country" not in d.columns:
        d["expected_date"] = pd.to_datetime(d[exp_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
        d["is_bank_holiday"] = False
        d["holiday_name"] = ""
        return d

    # Normalize expected_date
    d["expected_date"] = pd.to_datetime(d[exp_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

    # Normalize holidays df
    if holidays is None or holidays.empty:
        d["is_bank_holiday"] = False
        d["holiday_name"] = ""
        return d

    h = holidays.copy()
    for col in ["country", "date", "name"]:
        if col not in h.columns:
            # force empty result if schema unexpected
            d["is_bank_holiday"] = False
            d["holiday_name"] = ""
            return d

    h["country"] = h["country"].astype(str).str.upper().str.strip()
    h["date"] = h["date"].astype(str).str.slice(0, 10)
    h["name"] = h["name"].fillna("").astype(str)

    d["country"] = d["country"].astype(str).str.upper().str.strip()

    joined = d.merge(
        h.rename(columns={"name": "holiday_name"}),
        how="left",
        left_on=["country", "expected_date"],
        right_on=["country", "date"],
    )

    matched_holiday = joined["date"].notna()
    joined["holiday_name"] = joined["holiday_name"].fillna("").astype(str).str.strip()
    blank_name = joined["holiday_name"].eq("") | joined["holiday_name"].str.lower().isin(["nan", "none", "nat"])
    joined.loc[matched_holiday & blank_name, "holiday_name"] = "Unnamed holiday"
    joined["is_bank_holiday"] = matched_holiday

    # keep it tidy
    if "date" in joined.columns:
        joined = joined.drop(columns=["date"])

    return joined