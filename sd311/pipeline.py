"""Scheduled pipeline.  `refresh` (daily): pull counts, backtest, forecast. `tickets` (weekly): train ticket models."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data, tickets, volume

DOCS = Path("docs")


def refresh(weeks: int = 52):
    daily = data.daily_counts("2020-01-01")
    y = daily.set_index("d").n.astype(float).asfreq("D").interpolate()
    # the API's newest day is usually partial: drop days after the last full day
    y = y.iloc[:-1]
    bt = volume.backtest(y, weeks=weeks)
    mt = volume.metrics(bt)
    iv = volume.intervals(bt)
    origin = y.index.max()
    fc = volume.live_forecast(y)
    fc["lo"], fc["hi"] = fc.pred * (1 + iv["lo"]), fc.pred * (1 + iv["hi"])
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_full_day": str(origin.date()), "total_rows_in_dataset": data.total_rows(),
        "history": [{"d": str(d.date()), "n": int(v)} for d, v in y.iloc[-120:].items()],
        "forecast": [{"d": str(r.target_date.date()), "pred": round(r.pred), "naive": round(r.naive),
                      "lo": round(r.lo), "hi": round(r.hi)} for r in fc.itertuples()],
        "backtest": {"weeks": weeks, "n_forecasts": int(len(bt)), "metrics": mt, "interval": iv,
                     "series": [{"d": str(r.target_date.date()), "h": int(r.h), "pred": round(r.pred), "actual": round(r.actual)}
                                for r in bt[bt.h == 1].itertuples()]},
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "forecast.json").write_text(json.dumps(out))
    print("WAPE model %.3f vs seasonal-naive %.3f; 80%% interval held-out coverage %.2f" %
          (mt["pred"]["wape"], mt["naive"]["wape"], iv["heldout_coverage"]))


def run_tickets(months: int = 5, suffix: str = "7"):
    end = pd.Timestamp.now('UTC').tz_localize(None).normalize() - pd.Timedelta(days=45)  # let tickets mature
    start = end - pd.Timedelta(days=30 * months)
    df = data.ticket_sample(str(start.date()), str(end.date()), suffix)
    split = end - pd.Timedelta(days=30)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "sample": {"start": str(start.date()), "end": str(end.date()), "rule": f"unique_key ends with '{suffix}'",
                      "rows": int(len(df)), "test_from": str(split.date())},
           "routing": tickets.routing(df, split), "resolution": tickets.resolution(df, split)}
    DOCS.mkdir(exist_ok=True)
    (DOCS / "tickets.json").write_text(json.dumps(out))
    r = out["routing"]
    print("routing acc (no complaint_type) %.3f, with %.3f, majority %.3f" %
          (r["no_complaint_type"]["accuracy"], r["with_complaint_type"]["accuracy"], r["majority_baseline_acc"]))
    print("resolution MAE h: gbm %.1f vs baseline %.1f" % (out["resolution"]["gbm"]["mae_h"], out["resolution"]["median_by_type_baseline"]["mae_h"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refresh", "tickets"])
    ap.add_argument("--weeks", type=int, default=52)
    a = ap.parse_args()
    refresh(a.weeks) if a.cmd == "refresh" else run_tickets()


if __name__ == "__main__":
    main()
