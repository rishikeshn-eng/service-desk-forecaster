"""7-day-ahead daily volume forecast with a rolling-origin backtest and calibrated intervals.

Direct multi-horizon model: for each (origin, h) predict target/mean7 with gradient boosting from features
that only use data up to the origin. The same builder produces training rows, backtest rows and the live forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor

H = 7
FEATS = ["h", "dow", "doy", "hol", "hol_prev", "hol_next", "r_same", "r_m28", "r_last", "naive_ratio"]


def build(y: pd.Series) -> pd.DataFrame:
    """All (origin, h) rows for origins in y, targets taken from y extended by H NaN days (live rows have NaN target)."""
    idx = pd.date_range(y.index.min(), y.index.max() + pd.Timedelta(days=H), freq="D")
    ye = y.reindex(idx)
    v = ye.to_numpy(dtype=float)
    n = len(y)
    hol = pd.Series(idx.isin(USFederalHolidayCalendar().holidays(idx.min() - pd.Timedelta(days=2), idx.max() + pd.Timedelta(days=2))).astype(int), index=idx).to_numpy()
    m7 = pd.Series(v[:n]).rolling(7).mean().to_numpy()
    m28 = pd.Series(v[:n]).rolling(28).mean().to_numpy()
    rows = []
    origins = np.arange(34, n)  # need 28 days of history
    for h in range(1, H + 1):
        t = origins + h
        ks = [k for k in (1, 2, 3, 4) if 7 * k >= h]
        same = np.nanmean(np.column_stack([v[t - 7 * k] for k in ks]), axis=1)
        d = pd.DataFrame({
            "origin": idx[origins], "target_date": idx[t], "h": h, "dow": idx[t].dayofweek, "doy": idx[t].dayofyear,
            "hol": hol[t], "hol_prev": hol[t - 1], "hol_next": np.where(t + 1 < len(idx), hol[np.minimum(t + 1, len(idx) - 1)], 0),
            "m7": m7[origins], "r_same": same / m7[origins], "r_m28": m28[origins] / m7[origins],
            "r_last": v[origins] / m7[origins], "naive_ratio": v[t - 7 * ks[0]] / m7[origins],
            "y": v[t]})
        d["target"] = d.y / d.m7
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _fit(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=250, random_state=0)
    return m.fit(train[FEATS], train["target"])


def _predict(m, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["origin", "target_date", "h", "y"]].copy()
    out["pred"] = m.predict(rows[FEATS]) * rows.m7.to_numpy()
    out["naive"] = rows.naive_ratio.to_numpy() * rows.m7.to_numpy()
    return out.rename(columns={"y": "actual"})


def backtest(y: pd.Series, weeks: int = 52, train_start: str = "2022-01-01") -> pd.DataFrame:
    """Weekly origins. Each model trains only on rows whose *target date* is <= the test origin."""
    y = y.loc[train_start:]
    frame = build(y).dropna(subset=["y"])
    test_origins = pd.date_range(end=y.index.max() - pd.Timedelta(days=H), periods=weeks, freq="7D")
    res = []
    for o in test_origins:
        tr = frame[frame.target_date <= o]
        res.append(_predict(_fit(tr), frame[frame.origin == o]))
    return pd.concat(res, ignore_index=True)


def live_forecast(y: pd.Series, train_start: str = "2022-01-01") -> pd.DataFrame:
    y = y.loc[train_start:]
    frame = build(y)
    origin = y.index.max()
    model = _fit(frame.dropna(subset=["y"]))
    return _predict(model, frame[frame.origin == origin])


def metrics(bt: pd.DataFrame) -> dict:
    out = {}
    for name in ("pred", "naive"):
        err = (bt[name] - bt["actual"]).abs()
        out[name] = {"mae": float(err.mean()), "wape": float(err.sum() / bt.actual.sum()),
                     "mape": float((err / bt.actual).mean())}
    out["by_horizon_wape"] = {int(h): float((g.pred - g.actual).abs().sum() / g.actual.sum()) for h, g in bt.groupby("h")}
    out["by_horizon_naive_wape"] = {int(h): float((g.naive - g.actual).abs().sum() / g.actual.sum()) for h, g in bt.groupby("h")}
    return out


def intervals(bt: pd.DataFrame, level: float = 0.8) -> dict:
    """Relative-error quantiles. Calibrate on the older half of backtest origins, report coverage on the newer half."""
    origins = sorted(bt.origin.unique())
    cal, ev = bt[bt.origin.isin(origins[: len(origins) // 2])], bt[bt.origin.isin(origins[len(origins) // 2:])]
    rel = lambda d: d.actual / d.pred - 1
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    lo, hi = np.quantile(rel(cal), q)
    cov = float(((ev.actual >= ev.pred * (1 + lo)) & (ev.actual <= ev.pred * (1 + hi))).mean())
    lo2, hi2 = np.quantile(rel(bt), q)  # production interval uses all backtest errors
    return {"level": level, "lo": float(lo2), "hi": float(hi2), "heldout_coverage": cov}
