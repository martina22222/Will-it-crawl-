"""Model and baseline builders shared by notebooks 01, 03 and 04."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import TIME_FEATS, GEO_FEATS, DIST_FEATS, RATE_FEATS

ALL_FEATS = TIME_FEATS + GEO_FEATS + DIST_FEATS + RATE_FEATS


def make_logreg(cols, C=1.0):
    num = [c for c in cols if c not in ("hour", "dow")]
    parts = []
    if "hour" in cols or "dow" in cols:
        parts.append(("onehot", OneHotEncoder(handle_unknown="ignore"), [c for c in ("hour", "dow") if c in cols]))
    parts.append(("scale", StandardScaler(), num))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=C, max_iter=500))


def make_tree(depth, leaf=1000):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0)


class AlwaysNo:
    """Baseline 1: never warn."""
    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class BusyRule:
    """Baseline 2: warn on weekdays 8-19h when the pickup zone is one of the busiest
    (top-k by number of training pickups). Everything is learned from the training month."""
    def __init__(self, top_k=20, hours=(8, 19)):
        self.top_k, self.hours = top_k, hours

    def fit(self, train_df):
        self.zones_ = set(train_df.PULocationID.value_counts().head(self.top_k).index)
        return self

    def predict_proba(self, df):
        h = df.tpep_pickup_datetime.dt.hour; d = df.tpep_pickup_datetime.dt.dayofweek
        p = ((d < 5) & h.between(*self.hours) & df.PULocationID.isin(self.zones_)).astype(float).values
        return np.column_stack([1 - p, p])
