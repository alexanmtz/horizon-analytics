import pandas as pd
from metrics import add_derived_columns

def ensure_derived(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    # Always keep derived columns in the session df_enriched
    derived = add_derived_columns(df, mapping)
    return derived