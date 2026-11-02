"""Cleaning and target construction for the 'Will this trip crawl?' project.

Everything is rebuilt from the raw files by `build_dataset()`; nothing is edited by hand.
Every rule is a named, counted step so the audit table in notebook 02 comes from the same code.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CRAWL_MPH = 6.0          # target: average speed under 6 mph
MIN_DURATION_MIN = 1.0   # shorter than this, speed is rounding noise
MAX_DURATION_MIN = 180.0 # longer than 3 h is a meter left running
MIN_DISTANCE_MI = 0.1    # under 0.1 mi (~160 m) the distance is not meaningful
MAX_SPEED_MPH = 80.0     # faster than this is impossible inside NYC
AIRPORT_ZONES = (1, 132, 138)   # Newark, JFK, LaGuardia
UNKNOWN_ZONES = (264, 265)
SPLIT = {5: "train", 6: "validation", 7: "test"}  # pickup month -> split


def is_key(df, columns):
    """True if no combination of values in `columns` appears twice: the columns identify the grain."""
    return not df.duplicated(subset=columns).any()


def join_checked(left, right, **kwargs):
    """Left join that refuses to change the number of rows (a changed count means the grain was wrong)."""
    joined = left.merge(right, how="left", **kwargs)
    assert len(joined) == len(left), f"join changed the rows: {len(left):,} to {len(joined):,}"
    return joined


def load_raw():
    trips = pd.read_parquet(RAW / "yellow_tripdata_2026_sample.parquet")
    zones = pd.read_csv(RAW / "taxi_zone_lookup.csv")
    return trips, zones


def add_zone_names(trips, zones):
    assert is_key(zones, ["LocationID"]), "zone table is not one row per LocationID"
    pu = zones.rename(columns={"LocationID": "PULocationID", "Borough": "pickup_borough",
                               "Zone": "pickup_zone"})[["PULocationID", "pickup_borough", "pickup_zone"]]
    do = zones.rename(columns={"LocationID": "DOLocationID", "Borough": "dropoff_borough",
                               "Zone": "dropoff_zone"})[["DOLocationID", "dropoff_borough", "dropoff_zone"]]
    out = join_checked(trips, pu, on="PULocationID")
    return join_checked(out, do, on="DOLocationID")


def add_target_pieces(trips):
    """Duration, speed and the raw crawl label. Only used to build the target, never as features."""
    t = trips.copy()
    t["duration_min"] = (t.tpep_dropoff_datetime - t.tpep_pickup_datetime).dt.total_seconds() / 60
    t["speed_mph"] = t.trip_distance / (t.duration_min / 60)
    return t


def _crawl_rate(d):
    """Share with speed < 6 mph among rows where a speed exists (NaN if none)."""
    s = d.speed_mph.replace([np.inf, -np.inf], np.nan).dropna()
    return float((s < CRAWL_MPH).mean()) if len(s) else np.nan


# Each rule: (name, decision, function(df) -> boolean mask of rows to DROP)
RULES = [
    ("pickup outside May-July 2026", "drop",
     lambda d: ~d.tpep_pickup_datetime.dt.month.isin([5, 6, 7]) | (d.tpep_pickup_datetime.dt.year != 2026)),
    ("duration <= 0 (drop-off not after pickup)", "drop", lambda d: d.duration_min <= 0),
    (f"duration < {MIN_DURATION_MIN:g} min", "drop", lambda d: (d.duration_min > 0) & (d.duration_min < MIN_DURATION_MIN)),
    (f"duration > {MAX_DURATION_MIN:g} min", "drop", lambda d: d.duration_min > MAX_DURATION_MIN),
    (f"distance < {MIN_DISTANCE_MI} mi (incl. zero)", "drop", lambda d: d.trip_distance < MIN_DISTANCE_MI),
    ("negative or zero fare / total amount", "drop", lambda d: (d.fare_amount <= 0) | (d.total_amount <= 0)),
    ("unknown pickup or drop-off zone (264, 265)", "drop",
     lambda d: d.PULocationID.isin(UNKNOWN_ZONES) | d.DOLocationID.isin(UNKNOWN_ZONES)),
    ("payment type 3 (no charge) or 4 (dispute) or 6 (voided)", "drop", lambda d: d.payment_type.isin([3, 4, 6])),
    (f"speed > {MAX_SPEED_MPH:g} mph", "drop", lambda d: d.speed_mph > MAX_SPEED_MPH),
]


def clean(trips, zones):
    """Return (clean_df, audit_table). Rules are applied in order; each row is counted
    against the first rule that removes it, so the counts add up."""
    trips = trips.rename(columns={"Airport_fee": "airport_fee"})   # fix: misnamed (inconsistent capital)
    t = add_target_pieces(add_zone_names(trips, zones))
    n0 = len(t)
    rows = []
    for name, decision, rule in RULES:
        mask = rule(t)
        rows.append({"problem": name, "decision": decision, "rows_removed": int(mask.sum()),
                     "positive_rate_of_removed": _crawl_rate(t.loc[mask])})
        t = t[~mask]
    # kept-but-flagged problems
    flags = {
        "flex fare trips (payment type 0)": t.payment_type == 0,
        "non-standard rate code (2-6, 99)": t.RatecodeID.isin([2, 3, 4, 5, 6, 99]),
        "airport pickup or drop-off": t.PULocationID.isin(AIRPORT_ZONES) | t.DOLocationID.isin(AIRPORT_ZONES),
        "passenger_count 0 or missing": t.passenger_count.fillna(0) == 0,
    }
    for name, m in flags.items():
        rows.append({"problem": name, "decision": "keep + flag", "rows_removed": 0,
                     "rows_affected": int(m.sum()),
                     "positive_rate_of_removed": float((t.loc[m, "speed_mph"] < CRAWL_MPH).mean())})
    audit = pd.DataFrame(rows)
    audit["share_of_raw_%"] = (audit.rows_removed / n0 * 100).round(2)
    t = t.copy()
    t["crawl"] = (t.speed_mph < CRAWL_MPH).astype(int)
    t["month"] = t.tpep_pickup_datetime.dt.month
    t["split"] = t.month.map(SPLIT)
    t["is_flex"] = (t.payment_type == 0).astype(int)
    return t.reset_index(drop=True), audit


def build_dataset():
    trips, zones = load_raw()
    df, audit = clean(trips, zones)
    return df, audit


def target_ladder(trips, zones, month=None):
    """Positive rate after each labelling decision (for PROBLEM.md / notebook 01).
    month=5 restricts to the training month so no later label is looked at."""
    t = add_target_pieces(add_zone_names(trips, zones))
    if month is not None:
        t = t[t.tpep_pickup_datetime.dt.month == month]
    steps = [("raw, speed defined only where duration > 0 and distance > 0",
              (t.duration_min > 0) & (t.trip_distance > 0))]
    mask = steps[0][1]
    for label, extra in [
        (f"+ duration between {MIN_DURATION_MIN:g} and {MAX_DURATION_MIN:g} min", t.duration_min.between(MIN_DURATION_MIN, MAX_DURATION_MIN)),
        (f"+ distance >= {MIN_DISTANCE_MI} mi", t.trip_distance >= MIN_DISTANCE_MI),
        (f"+ speed <= {MAX_SPEED_MPH:g} mph", t.speed_mph <= MAX_SPEED_MPH),
    ]:
        mask = mask & extra
        steps.append((label, mask))
    out = [{"step": "all rows, naive (speed < 6 incl. zero distance, <=0 duration -> inf/NaN as not crawl)",
            "rows": len(t), "positive_rate": float((t.speed_mph < CRAWL_MPH).mean())}]
    for label, m in steps:
        out.append({"step": label, "rows": int(m.sum()), "positive_rate": float((t.loc[m, "speed_mph"] < CRAWL_MPH).mean())})
    return pd.DataFrame(out)
