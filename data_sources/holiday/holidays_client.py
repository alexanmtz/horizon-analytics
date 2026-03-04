import pandas as pd
from data_sources.http_client import fetch_json

BASE_URL = "https://openholidaysapi.org/PublicHolidays"


def _extract_holiday_name(h: dict) -> str:
    """
    OpenHolidays API commonly returns:
      name: [{"language":"EN","text":"..."} , ...]
    but can also be dict or string in some cases.
    """
    n = h.get("name")

    # Most common: list of localized entries
    if isinstance(n, list) and n:
        en = next((x for x in n if (x.get("language") or "").upper() == "EN"), None)
        pick = en or n[0]
        return (pick.get("text") or "").strip()

    # Sometimes dict
    if isinstance(n, dict):
        val = n.get("text") or n.get("en")
        if val:
            return str(val).strip()
        # fallback: first value
        try:
            return str(next(iter(n.values()))).strip()
        except Exception:
            return ""

    # Rare: string
    if isinstance(n, str):
        return n.strip()

    return ""


def fetch_holidays(country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch public holidays for a country in [start_date, end_date].

    start_date/end_date: YYYY-MM-DD
    Returns DataFrame: country, date, name
    """
    params = {
        "countryIsoCode": country,
        "languageIsoCode": "EN",
        "validFrom": start_date,
        "validTo": end_date,
    }
    data = fetch_json(
        BASE_URL,
        params=params,
        headers={"accept": "application/json"},
        timeout=20,
    )

    rows = []
    for h in data:
        d = (h.get("startDate") or h.get("date") or "")[:10]
        if not d:
            continue

        name = _extract_holiday_name(h)
        rows.append({"country": country, "date": d, "name": name})

    return pd.DataFrame(rows)