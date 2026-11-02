"""Leak-free features for the crawl model.

Allowed at pickup: pickup time, pickup zone, drop-off zone, and the *planned* distance of that
route. The raw files hold only the *measured* distance, so the planned distance is estimated
from history: the median measured distance of past trips between the same two zones,
learned from the training month only (`ZoneHistory`).
Forbidden (never selected): drop-off time, duration, speed, trip_distance, RatecodeID,
payment_type, fare/tip/toll/total amounts, passenger_count.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

AIRPORTS = (1, 132, 138)
KEY_COLS = ["tpep_pickup_datetime", "PULocationID", "DOLocationID",
            "pickup_borough", "dropoff_borough"]
FORBIDDEN = ["tpep_dropoff_datetime", "duration_min", "speed_mph", "trip_distance", "RatecodeID",
             "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
             "total_amount", "improvement_surcharge", "congestion_surcharge", "Airport_fee", "airport_fee",
             "cbd_congestion_fee", "passenger_count", "crawl"]

TIME_FEATS = ["hour", "dow", "is_weekend", "is_rush"]
GEO_FEATS = ["same_zone", "same_borough", "airport_pu", "airport_do",
             "pu_manhattan", "do_manhattan", "pu_queens", "pu_brooklyn"]
DIST_FEATS = ["est_dist", "log_est_dist"]
RATE_FEATS = ["pu_rate", "do_rate", "pair_rate"]


def base_features(df):
    """Deterministic features from pickup time and the two zones only."""
    t = df.tpep_pickup_datetime
    f = pd.DataFrame(index=df.index)
    f["hour"] = t.dt.hour
    f["dow"] = t.dt.dayofweek
    f["is_weekend"] = (f.dow >= 5).astype(int)
    f["is_rush"] = ((f.dow < 5) & f.hour.between(7, 19)).astype(int)
    f["same_zone"] = (df.PULocationID == df.DOLocationID).astype(int)
    f["same_borough"] = (df.pickup_borough == df.dropoff_borough).astype(int)
    f["airport_pu"] = df.PULocationID.isin(AIRPORTS).astype(int)
    f["airport_do"] = df.DOLocationID.isin(AIRPORTS).astype(int)
    f["pu_manhattan"] = (df.pickup_borough == "Manhattan").astype(int)
    f["do_manhattan"] = (df.dropoff_borough == "Manhattan").astype(int)
    f["pu_queens"] = (df.pickup_borough == "Queens").astype(int)
    f["pu_brooklyn"] = (df.pickup_borough == "Brooklyn").astype(int)
    f["PULocationID"] = df.PULocationID.values
    f["DOLocationID"] = df.DOLocationID.values
    return f


class ZoneHistory(BaseEstimator, TransformerMixin):
    """Learns from the training rows only:
       est_dist  - median measured distance of earlier trips on the same zone pair
                   (fallbacks: pickup-zone median, then global median) = planned-distance proxy
       pu_rate / do_rate / pair_rate - smoothed crawl rate of the pickup zone, drop-off zone, pair.
    `fit_transform` uses 5-fold out-of-fold values so a training trip never sees its own label
    or distance; `transform` (validation, test) uses the tables fitted on the full training set.
    `transform` never reads trip_distance or crawl; tests/leak_check in notebook 03 prove it."""

    def __init__(self, smooth_zone=50, smooth_pair=20, n_splits=5, seed=0):
        self.smooth_zone, self.smooth_pair, self.n_splits, self.seed = smooth_zone, smooth_pair, n_splits, seed

    # ---- learning the tables --------------------------------------------------
    def _tables(self, X, y, dist):
        d = pd.DataFrame({"pu": X.PULocationID.values, "do": X.DOLocationID.values,
                          "y": np.asarray(y), "dist": np.asarray(dist)})
        T = {"prior": d.y.mean(), "gdist": d.dist.median()}
        T["pu_dist"] = d.groupby("pu").dist.median()
        pair = d.groupby(["pu", "do"]).agg(n=("y", "size"), s=("y", "sum"), dist=("dist", "median"))
        T["pair"] = pair
        z = d.groupby("pu").y.agg(["size", "sum"]); T["pu_rt"] = z
        z = d.groupby("do").y.agg(["size", "sum"]); T["do_rt"] = z
        return T

    def _apply(self, X, T):
        pu, do = X.PULocationID.values, X.DOLocationID.values
        idx = pd.MultiIndex.from_arrays([pu, do])
        pair = T["pair"].reindex(idx)
        est = pair["dist"].values
        est = np.where(np.isnan(est), pd.Series(pu).map(T["pu_dist"]).values, est)
        est = np.where(np.isnan(est), T["gdist"], est)
        prior = T["prior"]

        def smooth(s, n, m, p):
            s, n = np.nan_to_num(s), np.nan_to_num(n)
            return (s + m * p) / (n + m)
        pr = T["pu_rt"].reindex(pu); dr = T["do_rt"].reindex(do)
        pu_rate = smooth(pr["sum"].values, pr["size"].values, self.smooth_zone, prior)
        do_rate = smooth(dr["sum"].values, dr["size"].values, self.smooth_zone, prior)
        pair_rate = smooth(pair["s"].values, pair["n"].values, self.smooth_pair, pu_rate)
        out = pd.DataFrame({"est_dist": est, "log_est_dist": np.log1p(est),
                            "pu_rate": pu_rate, "do_rate": do_rate, "pair_rate": pair_rate}, index=X.index)
        return out[DIST_FEATS + RATE_FEATS]

    # ---- sklearn API ----------------------------------------------------------
    def fit(self, X, y, trip_distance=None):
        self.tables_ = self._tables(X, y, X["trip_distance"] if trip_distance is None else trip_distance)
        return self

    def transform(self, X):
        return self._apply(X, self.tables_)

    def fit_transform(self, X, y, trip_distance=None):
        dist = X["trip_distance"] if trip_distance is None else trip_distance
        self.fit(X, y, dist)
        out = pd.DataFrame(index=X.index, columns=DIST_FEATS + RATE_FEATS, dtype=float)  # same order as _apply
        y = np.asarray(y); dist = np.asarray(dist)
        for tr, va in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(X):
            T = self._tables(X.iloc[tr], y[tr], dist[tr])
            out.iloc[va] = self._apply(X.iloc[va], T)[DIST_FEATS + RATE_FEATS].values
        return out


def build_matrix(train, others, feature_sets=None):
    """train: dataframe of the training month. others: dict name -> dataframe.
    Returns dict name -> feature DataFrame (all candidate columns), fitted on train only."""
    zh = ZoneHistory()
    cols = ["PULocationID", "DOLocationID", "trip_distance"]
    Xtr_hist = zh.fit_transform(train[cols], train["crawl"])
    out = {"train": pd.concat([base_features(train), Xtr_hist], axis=1)}
    for name, d in others.items():
        out[name] = pd.concat([base_features(d), zh.transform(d[cols])], axis=1)
    return out, zh
