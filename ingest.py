from pathlib import Path
from io import StringIO
import csv
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


def load_table_from_text(raw_text: str) -> pd.DataFrame:
    """
    Loads pasted tabular text (CSV/TSV/semicolon-delimited) and normalizes columns.
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty input. Paste CSV/TSV data with a header row.")

    sep = _detect_separator(text)
    df = pd.read_csv(StringIO(text), sep=sep, engine="python")

    if df.empty:
        raise ValueError("No rows detected in pasted data.")

    df.columns = [normalize_col(c) for c in df.columns]
    return df


def _detect_separator(text: str) -> str | None:
    sample = "\n".join(text.splitlines()[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"
        if ";" in sample:
            return ";"
        if "|" in sample:
            return "|"
        return ","


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