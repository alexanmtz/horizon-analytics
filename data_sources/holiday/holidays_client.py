import pandas as pd
from datetime import date, timedelta
from clients.http_client import fetch_json

BASE_URL = "https://openholidaysapi.org/PublicHolidays"


def _extract_holiday_name(h: dict, preferred_languages: tuple[str, ...] = ("EN",)) -> str:
    """
    OpenHolidays API commonly returns:
      name: [{"language":"EN","text":"..."} , ...]
    but can also be dict or string in some cases.
    """
    n = h.get("name")

    # Most common: list of localized entries
    if isinstance(n, list) and n:
        dict_items = [x for x in n if isinstance(x, dict)]

        # Prefer caller-defined languages if they have non-empty text
        normalized_preferred = [lang.upper() for lang in preferred_languages if str(lang).strip()]
        for preferred in normalized_preferred:
            for item in dict_items:
                if (item.get("language") or "").upper() == preferred:
                    text = str(item.get("text") or "").strip()
                    if text:
                        return text

        # Fallback: first non-empty localized text
        for item in dict_items:
            text = str(item.get("text") or "").strip()
            if text:
                return text

        # Last resort: stringified first item if possible
        try:
            return str(n[0]).strip()
        except Exception:
            return ""

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

    def _parse_day(value: str | None) -> date | None:
        if not value:
            return None
        text = str(value)[:10]
        try:
            return pd.to_datetime(text, errors="coerce", utc=True).date()
        except Exception:
            return None

    for h in data:
        start_day = _parse_day(h.get("startDate") or h.get("date"))
        end_day = _parse_day(h.get("endDate") or h.get("startDate") or h.get("date"))

        if start_day is None:
            continue

        if end_day is None or end_day < start_day:
            end_day = start_day

        preferred_languages = ("PT", "EN") if str(country).strip().upper() == "BR" else ("EN",)
        name = _extract_holiday_name(h, preferred_languages=preferred_languages)
        day = start_day
        while day <= end_day:
            rows.append({"country": country, "date": day.isoformat(), "name": name})
            day += timedelta(days=1)

    return pd.DataFrame(rows)