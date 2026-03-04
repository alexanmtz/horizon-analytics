import pandas as pd

from data_sources.holiday.holidays_client import fetch_holidays
from data_sources.holiday.enrich_holidays import enrich_with_holidays


def connect_datasource(df: pd.DataFrame, mapping: dict, source_id: str) -> tuple[pd.DataFrame, dict]:
    if source_id != "openholidays":
        raise ValueError(f"Unsupported datasource: {source_id}")

    if "country" not in df.columns:
        raise ValueError("Dataset has no country column.")

    exp_col = mapping.get("expected_arrival_at")
    if not exp_col or exp_col not in df.columns:
        raise ValueError("Missing expected_arrival_at mapping.")

    exp = pd.to_datetime(df[exp_col], errors="coerce", utc=True)
    exp = exp.dropna()
    if exp.empty:
        raise ValueError("No valid expected_arrival_at timestamps.")

    start_date = exp.min().strftime("%Y-%m-%d")
    end_date = exp.max().strftime("%Y-%m-%d")

    countries = sorted([c for c in df["country"].dropna().unique().tolist() if isinstance(c, str) and c.strip()])
    all_holidays = []
    skipped = []

    for country in countries:
        try:
            all_holidays.append(fetch_holidays(country, start_date, end_date))
        except Exception:
            skipped.append(country)

    holidays_df = pd.concat(all_holidays, ignore_index=True) if all_holidays else pd.DataFrame(columns=["country", "date", "name"])
    enriched = enrich_with_holidays(df, holidays_df, mapping)

    meta = {
        "source_id": source_id,
        "base_url": "https://openholidaysapi.org/PublicHolidays",
        "country_count": len(countries),
        "date_range": [start_date, end_date],
        "holidays_rows": int(len(holidays_df)),
        "skipped_countries": skipped,
        "new_columns": [c for c in ["expected_date", "is_bank_holiday", "holiday_name"] if c in enriched.columns],
    }

    return enriched, meta
