from pathlib import Path
import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    """
    Loads CSV/XLSX and normalizes column names.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use CSV or XLSX.")

    df.columns = [normalize_col(c) for c in df.columns]
    return df


def normalize_col(name: str) -> str:
    """
    Normalizes column names to snake_case-ish:
    - spaces/dashes/slashes -> underscore
    - remove weird characters
    - lowercase
    """
    s = str(name).strip()
    s = s.replace(" ", "_").replace("-", "_").replace("/", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.lower()