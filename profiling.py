import pandas as pd


def profile_df(df: pd.DataFrame) -> dict:
    """
    Lightweight dataset profiling.

    Returns:
    {
        "rows": int,
        "cols": int,
        "columns": [
            {
                "name": str,
                "dtype": str,
                "null_pct": float,
                "nunique": int
            },
            ...
        ]
    }
    """

    column_profiles = []

    for col in df.columns:
        series = df[col]

        null_pct = float(series.isna().mean() * 100.0)
        nunique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        column_profiles.append(
            {
                "name": col,
                "dtype": dtype,
                "null_pct": null_pct,
                "nunique": nunique,
            }
        )

    # Sort columns by usefulness:
    # 1. fewer nulls first
    # 2. more unique values first
    column_profiles_sorted = sorted(
        column_profiles,
        key=lambda x: (x["null_pct"], -x["nunique"]),
    )

    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": column_profiles_sorted,
    }