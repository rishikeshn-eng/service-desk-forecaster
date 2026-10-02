"""Agency routing and resolution-time models on a stated ticket sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

ROUTE_TEXT = ["descriptor", "location_type", "address_type", "borough", "open_data_channel_type"]


def _text(df: pd.DataFrame, cols) -> pd.Series:
    return df[cols].fillna("na").astype(str).apply(lambda r: " ".join(f"{c}={v.replace(' ', '_')}" for c, v in zip(cols, r)), axis=1)


def routing(df: pd.DataFrame, split_date) -> dict:
    """Predict `agency`. Main model must NOT see complaint_type (the operator's pick that practically fixes the
    agency); a with-complaint_type model is reported as the ceiling."""
    top = df.agency.value_counts()
    keep = top[top >= 30].index
    d = df[df.agency.isin(keep)]
    tr, te = d[d.created_date < split_date], d[d.created_date >= split_date]
    out = {"n_train": len(tr), "n_test": len(te), "n_agencies": len(keep),
           "majority_baseline_acc": float((te.agency == tr.agency.mode()[0]).mean())}
    for name, cols in (("no_complaint_type", ROUTE_TEXT), ("with_complaint_type", ROUTE_TEXT + ["complaint_type"])):
        vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 1), min_df=2)
        clf = make_pipeline(vec, LogisticRegression(max_iter=300, C=5.0))
        clf.fit(_text(tr, cols), tr.agency)
        pred = clf.predict(_text(te, cols))
        out[name] = {"accuracy": float(accuracy_score(te.agency, pred)),
                     "macro_f1": float(f1_score(te.agency, pred, average="macro"))}
        if name == "no_complaint_type":
            wrong = te.assign(pred=pred)[pred != te.agency]
            out["top_confusions"] = [{"true": a, "pred": b, "n": int(n)} for (a, b), n in
                                     wrong.groupby(["agency", "pred"]).size().sort_values(ascending=False).head(5).items()]
    return out


RES_CAT = ["complaint_type", "agency", "borough", "location_type", "open_data_channel_type", "incident_zip", "descriptor"]


def resolution(df: pd.DataFrame, split_date, max_days: int = 90) -> dict:
    """Predict hours to close (median-style: absolute-error loss on log1p hours).

    Tickets must have closed; those closing after max_days are dropped (rare, and right-censoring makes the tail
    unreliable). Rare category levels are pooled into "other" using the training set only (HGB allows <=255).
    Baseline: per-complaint_type median.
    """
    d = df.dropna(subset=["closed_date"]).copy()
    d["hours"] = (d.closed_date - d.created_date).dt.total_seconds() / 3600
    d = d[(d.hours >= 0) & (d.hours <= max_days * 24)]
    d["hour"], d["dow"] = d.created_date.dt.hour, d.created_date.dt.dayofweek
    tr, te = d[d.created_date < split_date], d[d.created_date >= split_date]

    def pooled(frame):
        out = {}
        for c in RES_CAT:
            top = set(tr[c].fillna("na").astype(str).value_counts().head(150).index)
            s = frame[c].fillna("na").astype(str)
            out[c] = s.where(s.isin(top), "other")
        return pd.DataFrame(out)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    Xtr = np.column_stack([enc.fit_transform(pooled(tr)), tr[["hour", "dow"]].values])
    Xte = np.column_stack([enc.transform(pooled(te)), te[["hour", "dow"]].values])
    cat_mask = [True] * len(RES_CAT) + [False, False]
    gbm = HistGradientBoostingRegressor(loss="absolute_error", categorical_features=cat_mask, max_iter=200,
                                        learning_rate=0.08, random_state=0)
    gbm.fit(Xtr, np.log1p(tr.hours))
    pred = np.expm1(gbm.predict(Xte))
    med = tr.groupby("complaint_type").hours.median()
    base = te.complaint_type.map(med).fillna(tr.hours.median())
    mae = lambda p: float(np.abs(p - te.hours).mean())
    medae = lambda p: float(np.median(np.abs(p - te.hours)))
    within = lambda p: float((np.abs(np.log1p(p) - np.log1p(te.hours)) < np.log(2)).mean())
    return {"n_train": len(tr), "n_test": len(te), "median_hours_test": float(te.hours.median()),
            "gbm": {"mae_h": mae(pred), "medae_h": medae(pred), "within_2x": within(pred)},
            "median_by_type_baseline": {"mae_h": mae(base), "medae_h": medae(base), "within_2x": within(base)},
            "excluded_over_max_days": int((df.closed_date.notna() & ((df.closed_date - df.created_date).dt.days > max_days)).sum())}
