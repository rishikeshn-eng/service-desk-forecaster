"""NYC Open Data (Socrata) access for the 311 Service Requests dataset (2020 to present)."""
from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
TICKET_COLS = ["unique_key", "created_date", "closed_date", "agency", "complaint_type", "descriptor",
               "location_type", "address_type", "borough", "incident_zip", "open_data_channel_type"]


def _get(params: dict, retries: int = 4) -> list[dict]:
    url = BASE + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "sd311/0.1"}
    if os.environ.get("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i * 2)


def daily_counts(since: str = "2020-01-01", by_agency: bool = False) -> pd.DataFrame:
    """Population-level daily request counts (server-side aggregation; no row download)."""
    sel = "date_trunc_ymd(created_date) as d, " + ("agency, " if by_agency else "") + "count(*) as n"
    grp = "d, agency" if by_agency else "d"
    rows = _get({"$select": sel, "$where": f"created_date >= '{since}T00:00:00'", "$group": grp,
                 "$order": "d", "$limit": 200000})
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    df["n"] = df["n"].astype(int)
    return df


def total_rows() -> int:
    return int(_get({"$select": "count(*) as n"})[0]["n"])


def ticket_sample(start: str, end: str, key_suffix: str = "7", page: int = 50000, cache_dir: str | None = "data") -> pd.DataFrame:
    """Closed-or-open tickets created in [start, end), deterministic ~10^-len(suffix) sample by unique_key suffix.

    Cached as parquet in `cache_dir` (set None to disable) because the API is slow for 100k+ rows.
    """
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"tickets_{start}_{end}_{key_suffix}.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)
    out = []
    off = 0
    while True:
        rows = _get({"$select": ",".join(TICKET_COLS),
                     "$where": f"created_date >= '{start}T00:00:00' AND created_date < '{end}T00:00:00' "
                               f"AND unique_key like '%{key_suffix}'",
                     "$order": "unique_key", "$limit": page, "$offset": off})
        out += rows
        if len(rows) < page:
            break
        off += page
    df = pd.DataFrame(out, columns=TICKET_COLS)
    for c in ("created_date", "closed_date"):
        df[c] = pd.to_datetime(df[c])
    if cache:
        df.to_parquet(cache)
    return df
