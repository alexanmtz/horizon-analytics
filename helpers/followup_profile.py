def followup_profile(df, mapping: dict) -> dict:
    cols = list(df.columns)
    return {
        "columns": cols,
        "mapping": mapping,
        "has_delay": "delay_hours" in cols or "delay_days" in cols,
        "has_transfer_time": "transfer_time_hours" in cols,
    }