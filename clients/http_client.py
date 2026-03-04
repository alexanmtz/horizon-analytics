import requests

def fetch_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 20):
    response = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()
