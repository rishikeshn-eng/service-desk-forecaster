import numpy as np
import pandas as pd

from sd311 import data, tickets, volume


def series(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    dow = np.array([1.1, 1.15, 1.1, 1.05, 1.0, 0.8, 0.8])[idx.dayofweek]
    return pd.Series(10000 * dow * rng.lognormal(0, 0.04, n), index=idx)


def test_features_use_only_past_data():
    y = series()
    f1 = volume.build(y)
    y2 = y.copy()
    y2.iloc[-3:] *= 5  # change only the newest days
    f2 = volume.build(y2)
    cut = y.index[-12]  # origins well before the edit
    a, b = f1[f1.origin <= cut], f2[f2.origin <= cut]
    cols = [c for c in volume.FEATS + ["m7"] if c != "y"]
    # features for those origins may depend on the target only through y/target columns, never the inputs
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def test_live_forecast_has_seven_days_after_last_observation():
    y = series()
    fc = volume.live_forecast(y)
    assert len(fc) == 7 and fc.target_date.min() == y.index.max() + pd.Timedelta(days=1)
    assert fc.pred.between(4000, 20000).all()


def test_backtest_beats_naive_on_weekly_pattern_and_intervals_cover():
    y = series(700)
    bt = volume.backtest(y, weeks=20, train_start="2022-01-01")
    m = volume.metrics(bt)
    assert m["pred"]["wape"] < 0.2 and m["pred"]["wape"] <= m["naive"]["wape"] * 1.1
    iv = volume.intervals(bt)
    assert 0.6 <= iv["heldout_coverage"] <= 1.0


def test_daily_counts_parsing(monkeypatch):
    monkeypatch.setattr(data, "_get", lambda p: [{"d": "2026-01-01T00:00:00.000", "n": "100"},
                                                  {"d": "2026-01-02T00:00:00.000", "n": "120"}])
    df = data.daily_counts()
    assert df.n.tolist() == [100, 120] and str(df.d.iloc[0].date()) == "2026-01-01"


def tiny_tickets(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    ct = rng.choice(["Noise", "Pothole", "Heat"], n)
    agency = np.array([{"Noise": "NYPD", "Pothole": "DOT", "Heat": "HPD"}[c] for c in ct])
    desc = np.array([{"Noise": "loud music", "Pothole": "pothole", "Heat": "no heat"}[c] for c in ct])
    created = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 60 * 24, n), unit="h")
    hrs = np.array([{"Noise": 2, "Pothole": 72, "Heat": 24}[c] for c in ct]) * rng.lognormal(0, 0.3, n)
    return pd.DataFrame({"created_date": created, "closed_date": created + pd.to_timedelta(hrs, unit="h"),
                         "agency": agency, "complaint_type": ct, "descriptor": desc, "location_type": "street",
                         "address_type": "ADDRESS", "borough": "BRONX", "open_data_channel_type": "PHONE",
                         "incident_zip": rng.choice(["10451", "10452", "10453"], n)})


def test_routing_and_resolution_models_learn_obvious_structure():
    df = tiny_tickets()
    split = pd.Timestamp("2026-02-15")
    r = tickets.routing(df, split)
    assert r["no_complaint_type"]["accuracy"] > 0.95 and r["majority_baseline_acc"] < 0.5
    s = tickets.resolution(df, split)
    # lognormal(0, 0.3) noise around per-type means: both models should be close, GBM no worse than 1.5x baseline
    assert s["gbm"]["mae_h"] < 1.5 * s["median_by_type_baseline"]["mae_h"] + 1
