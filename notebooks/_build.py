"""Generates the four notebooks (run from repo root: python notebooks/_build.py)."""
import nbformat as nbf
from pathlib import Path

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n

SETUP = '''import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.clean import *
from src.features import *
from src.models import *
from src.evaluate import *
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
FIG = Path("../reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
df, audit = build_dataset()
train, val, test = [df[df.split == s].reset_index(drop=True) for s in ("train", "validation", "test")]
'''

# ------------------------------------------------------------------ 01
n1 = nb([
("md", "# 01 - Frame: will this trip crawl?\n\n**Question (one sentence):** at the moment a yellow-taxi trip starts, can we tell from the time and the two zones whether its average speed will end up under 6 mph, so a dispatcher can warn the passenger?\n\nThis notebook fixes the target, its labelling decisions and the baseline ladder. See `PROBLEM.md` for the written version."),
("code", SETUP),
("md", "## 1. The target and what can be labelled\nSpeed = `trip_distance / duration`. A trip is positive if speed < 6 mph. Speed is meaningless for zero distances, zero/negative durations and tiny durations, so those trips cannot be labelled. The table shows the positive rate after each decision, computed on **May only (the training month)** so that no June or July label is looked at while we define the target. A second table on all months follows, only to show the rule choices do not depend on the month."),
("code", "trips_raw, zones = load_raw()\nladder = target_ladder(trips_raw, zones, month=5)\nladder['positive_rate'] = (ladder.positive_rate*100).round(2)\nall_months = target_ladder(trips_raw, zones)\nall_months['positive_rate'] = (all_months.positive_rate*100).round(2)\nprint('MAY ONLY (used for decisions)'); print(ladder.to_string()); print('\\nALL MONTHS (comparison only; the steps barely change the rate in any month)'); print(all_months.to_string())"),
("md", "**Reading it.** A naive label counts zero-distance trips (speed 0) as crawls and inflates the rate; the decisions below remove them. Each rule and its reason:\n\n| Decision | Why |\n|---|---|\n| duration must be > 0 | drop-off not after pickup: speed is undefined |\n| duration 1-180 min | under 1 min the speed is rounding noise; over 3 h the meter was left running |\n| distance >= 0.1 mi | under ~160 m the distance is not a real trip, and speed 0 would fake a crawl |\n| speed <= 80 mph | impossible in NYC, a measurement error |\n\nThe full cleaning (negative fares, unknown zones, payment types) is in notebook 02."),
("md", "## 2. The moment of prediction\nAt pickup we know the **pickup time, the pickup zone, the drop-off zone** and the **planned route distance**. We do **not** know the duration, drop-off time, measured distance, rate code, payment type, any amount, tip or toll. `src/features.py` holds the `FORBIDDEN` list and notebook 03 tests it.\n\n**Why the planned distance is allowed:** the target is distance/duration. A planned (routed) distance is known when the ride is booked, before any driving happens, so it is a legitimate input; it is not the target's numerator measured after the fact. The files only contain the *measured* distance, so we estimate the plan as the median measured distance of *earlier* trips between the same two zones, learned from May only."),
("md", "## 3. The baseline ladder (scored on June)\n1. Always predict 'does not crawl'.\n2. A dispatcher rule: weekdays 8-19h, pickup in one of the 20 busiest pickup zones (busiest = most May pickups).\n3. A decision tree two levels deep."),
("code", '''M, zh = build_matrix(train, {"val": val})
ytr, yva = train.crawl.values, val.crawl.values
b1 = AlwaysNo().predict_proba(val)[:, 1]
rule = BusyRule().fit(train); b2 = rule.predict_proba(val)[:, 1]
tree2 = make_tree(2).fit(M["train"][ALL_FEATS], ytr); b3 = tree2.predict_proba(M["val"][ALL_FEATS])[:, 1]
rows = {}
for name, p, thr in [("1 always 'no crawl'", b1, 0.5), ("2 busy-hours rule", b2, 0.5), ("3 depth-2 tree", b3, 0.30)]:
    rows[name] = metrics(yva, p, thr)
pd.DataFrame(rows).T.round(3)'''),
("code", '''from sklearn.tree import export_text
print(export_text(tree2, feature_names=ALL_FEATS))
print("Rule zones:", sorted(rule.zones_))'''),
("md", "**Notes.** Baseline 1 has accuracy ~82% yet catches nothing (recall 0), which is why accuracy alone is never reported. Baselines 2 and 3 are the bar the models must clear. The depth-2 tree uses a probability threshold of 0.30 (any value works for 'warn if leaf rate is high'; the real threshold is chosen from costs in notebook 04)."),
])

# ------------------------------------------------------------------ 02
n2 = nb([
("md", "# 02 - Clean, split and analyse\n\nAll cleaning lives in `src/clean.py::build_dataset()` and rebuilds the data from the raw files. This notebook shows the audit, the split and the analysis required for Part C. **Exploration uses May (training) only**, except the month-to-month stability check, which uses only the label rate."),
("code", SETUP),
("md", "## 0. Grain: what one row stands for, and joins that cannot multiply rows\nA table's *grain* is what one row stands for. We test it instead of assuming it, and every join goes through `join_checked`, which stops if the row count changes."),
("code", """trips_raw, zones_raw = load_raw()
grain = pd.DataFrame([
    {"table": "trips", "one row is": "one yellow-taxi trip (there is no ID column)", "key tested": "all 21 columns together", "holds?": is_key(trips_raw, list(trips_raw.columns))},
    {"table": "zones", "one row is": "one taxi zone", "key tested": "LocationID", "holds?": is_key(zones_raw, ["LocationID"])},
])
print(grain.to_string(index=False))
merged = add_zone_names(trips_raw, zones_raw)       # two checked joins (pickup zone, drop-off zone)
print(f"rows before the zone joins: {len(trips_raw):,}  after both joins: {len(merged):,}")
assert trips_raw.PULocationID.isin(zones_raw.LocationID).all() and trips_raw.DOLocationID.isin(zones_raw.LocationID).all()
print("every pickup and drop-off zone exists in the lookup")
zones_raw[zones_raw.duplicated("Zone", keep=False)][["LocationID", "Borough", "Zone"]]"""),
("md", "**Reading it.** Both keys hold, and the join keeps exactly 600,000 rows. Five zone IDs share names (56 and 57 are both 'Corona'; 103, 104, 105 are all 'Governor's Island/Ellis Island/Liberty Island'). That is harmless, because we join on the ID, never on the name. There is one trips table and one lookup, so the grain problem of many-to-many joins does not arise, but we still test it."),
("md", "## 1. Audit and decisions (findings table)\nEach rule removes the rows it names (counted in order, so the removals add up). `positive_rate_of_removed` is the crawl share among removed rows where a speed exists. Rows kept but flagged are listed at the bottom."),
("code", "a = audit.copy(); a['positive_rate_of_removed'] = a.positive_rate_of_removed.round(3)\nprint(f'raw rows: 600,000 -> clean rows: {len(df):,}  ({100*len(df)/600000:.1f}% kept)')\na"),
("md", '''| Problem | Decision and defence |
|---|---|
| Pickup outside May-July | 2 April rows; outside the split, drop |
| Duration <= 0, < 1 min, > 3 h | speed undefined or noise; drop (these are 79-94% 'crawls' when defined, which would pollute the label) |
| Distance < 0.1 mi / zero | cannot compute a meaningful speed; every one would be a fake crawl; drop |
| Negative / zero fare or total | refunds, corrections, not real trips; drop |
| Unknown zones 264/265 | no zone to learn from or map; drop (0.6%) |
| Payment type 3 / 4 / 6 | no charge, dispute, voided: not completed ordinary fares; drop (0.4%). Type 6 does not occur in the sample |
| Speed > 80 mph | impossible; drop (102 rows) |
| **Flex fare (type 0)** | **kept** and flagged (`is_flex`). They are a quarter of trips and real, measured trips; the target needs only distance and duration, which they have. They have no passenger count or rate code, but we use neither. Dropping them would change the population the dispatcher serves |
| Non-standard rate codes (2-6, 99) | **kept**; real trips (JFK flat fare, negotiated...). The rate code is set at the trip end so is not used as a feature |
| Airport trips | kept; analysed separately below |'''),
("md", "### The six-category checklist (counts on the raw data, each with a decision and a reason)\nThe ordered audit above counts each row once. This table lists every problem by category, so each of the six checks is visibly done."),
("code", """r2 = add_target_pieces(trips_raw); miss = trips_raw.isna().sum(); n_flex = int((trips_raw.payment_type == 0).sum())
months_ok = trips_raw.tpep_pickup_datetime.dt.month.isin([5, 6, 7]) & (trips_raw.tpep_pickup_datetime.dt.year == 2026)
rows = [
 ("Missing", f"{int(miss['passenger_count']):,} trips lack passenger count, rate code, store flag, congestion and airport fee; exactly the {n_flex:,} flex-fare trips", "Flag (is_flex)", "None of those columns is a feature; the target only needs distance and duration"),
 ("Missing", f"request_source is empty for {int(miss['request_source']):,} trips", "Flag", "Unknown meaning, never used"),
 ("Missing", "zone 264 has borough 'N/A' (read as missing) and zone 265 'Unknown'", "Drop their trips", "No zone to learn from or map"),
 ("Impossible", f"{int((r2.duration_min <= 0).sum()):,} trips where drop-off is not after pickup", "Drop", "Speed is undefined"),
 ("Impossible", f"{int(((r2.duration_min > 0) & (r2.speed_mph > 80)).sum()):,} trips above 80 mph among trips with a positive duration (raw count)", "Drop", "Not possible inside New York"),
 ("Out of range", f"{int((trips_raw.trip_distance < 0.1).sum()):,} trips under 0.1 mile (incl. zero)", "Drop", "Distance is not meaningful and would fake a crawl"),
 ("Out of range", f"{int((r2.duration_min > 180).sum()):,} trips over 3 hours; {int((r2.duration_min.between(0, 1, inclusive='neither')).sum()):,} under 1 minute (raw counts)", "Drop", "Meter left running / rounding noise"),
 ("Out of range", f"{int((~months_ok).sum()):,} trips picked up outside May-July 2026", "Drop", "Outside the split"),
 ("Out of range", f"{int(((trips_raw.fare_amount <= 0) | (trips_raw.total_amount <= 0)).sum()):,} trips with zero or negative fare or total", "Drop", "Refunds or corrections, not ordinary trips"),
 ("Inconsistent", f"{int(trips_raw.payment_type.isin([3, 4]).sum()):,} no-charge or disputed trips (payment 3, 4)", "Drop", "Not ordinary paid trips; their crawl rate looks abnormal"),
 ("Inconsistent", f"{int(trips_raw.RatecodeID.isin([2, 3, 4, 5, 6, 99]).sum()):,} trips with a non-standard rate code", "Keep + flag", "Real trips; the code is only set at the end so it is not a feature"),
 ("Duplicated", f"{int(trips_raw.duplicated().sum())} exact duplicate trips", "None needed", "Tested with duplicated()"),
 ("Duplicated", "5 zone IDs share 2 names (see above)", "None needed", "We join on the ID, not the name"),
 ("Misnamed", "column 'Airport_fee' is capitalised unlike the others", "Fix", "Renamed to airport_fee in clean(); it is not used as a feature anyway"),
]
pd.DataFrame(rows, columns=["category", "finding", "decision", "reason"])"""),
("md", "### Cleaning fixes facts; it never learns\n| Step | Fact or learned? |\n|---|---|\n| Drop duration <= 0, < 1 min, > 3 h, distance < 0.1 mi, speed > 80 mph | **Fact**: fixed physical limits, the same answer on any sample. (We chose the limits after looking at the whole file, but they rest on physics, not on the labels.) |\n| Drop negative fares, unknown zones 264/265, payment types 3 and 4 | **Fact**: a property of each row |\n| Rename `Airport_fee`, join zone names | **Fact** |\n| Fill missing passenger count or rate code with a typical value | **Not done**: that would be learning, and we do not use those columns |\n| Planned-distance estimate, zone and zone-pair crawl rates | **Learned**: they live in `ZoneHistory` (`src/features.py`), fitted on May only after the split, out-of-fold inside May |"),
("md", "## 2. The split (by pickup month)"),
("code", "df.groupby('split').agg(trips=('crawl','size'), positive_rate=('crawl','mean')).loc[['train','validation','test']].round(4)"),
("md", "### Why a time split\n**Who or when will the model be used on?** On *future* trips in the same city by the same taxi fleet, with zones and zone pairs it has already seen. So the split is **by time** (May train, June validation, July test), as the course sets it. A grouped split (new people or new shops) does not apply: the data has no driver or passenger ID, and zone pairs repeat by design. Zones and pairs that appear in July but not in May fall back to broader averages in `ZoneHistory`.\n\n**Disclosure on exploring the target.** The audit, the cleaning and the structural checks run on all rows and never look at the label. Exploration of the label starts below and uses **May only**, with two exceptions that we state openly: the month-to-month stability check (it needs all three months but looks only at the label rate, and no modelling choice depends on it), and the target ladder in notebook 01, which now uses May only. While we were first developing the project, the label pattern was also looked at on all months, so the feature ideas are re-derived from the May charts below (see the table at the end)."),
("md", "## 3. Positive rate overall, by hour and day of week (May)"),
("code", '''train['hour'] = train.tpep_pickup_datetime.dt.hour; train['dow'] = train.tpep_pickup_datetime.dt.dayofweek
print(f"May positive rate: {train.crawl.mean():.3%}")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
train.groupby('hour').crawl.mean().plot(kind='bar', ax=ax[0], color='#c0392b'); ax[0].set_title('Crawl rate by pickup hour'); ax[0].set_ylabel('share of trips < 6 mph')
train.groupby('dow').crawl.mean().rename(dict(enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))).plot(kind='bar', ax=ax[1], color='#2c3e50'); ax[1].set_title('by day of week')
plt.tight_layout(); plt.savefig(FIG/'rate_by_hour_dow.png', dpi=130); plt.show()
heat = train.pivot_table(index='dow', columns='hour', values='crawl', aggfunc='mean')
plt.figure(figsize=(12, 3.2)); plt.imshow(heat.values, aspect='auto', cmap='Reds'); plt.colorbar(label='crawl rate')
plt.yticks(range(7), ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']); plt.xticks(range(24)); plt.xlabel('hour'); plt.title('Crawl rate, hour x day (May)')
plt.tight_layout(); plt.savefig(FIG/'heat_hour_dow.png', dpi=130); plt.show()'''),
("md", "**Four-step rhythm for this chart (team: write steps 1 and 3 yourselves; step 1 *before* you rerun the cell)**\n1. What we expected: ______\n2. The chart (above)\n3. What it shows, in one sentence: ______\n4. Feature it suggests: `hour`, `dow`, `is_weekend`, `is_rush`"),
("md", "## 4. Positive rate by pickup zone: the map"),
("code", '''import geopandas as gpd
shp = gpd.read_file("../data/raw/taxi_zones/taxi_zones.shp")     # coordinates in feet (EPSG:2263), fine for a map
z = train.groupby('PULocationID').crawl.agg(['size', 'mean']).rename(columns={'size': 'n', 'mean': 'rate'})
m = shp.merge(z, left_on='LocationID', right_index=True, how='left')
m['rate_shown'] = m.rate.where(m.n >= 100)          # grey out zones with < 100 pickups: too noisy
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
m.plot(column='rate_shown', cmap='Reds', legend=True, legend_kwds={'label': 'share of trips < 6 mph'}, missing_kwds={'color': '#dddddd'}, ax=ax[0], edgecolor='white', linewidth=.2)
ax[0].set_title('Crawl rate by pickup zone, May (grey: < 100 pickups)')
m.plot(column='rate_shown', cmap='Reds', missing_kwds={'color': '#dddddd'}, ax=ax[1], edgecolor='white', linewidth=.2)
ax[1].set_xlim(975000, 1010000); ax[1].set_ylim(185000, 260000); ax[1].set_title('Zoom: Manhattan')
for a_ in ax: a_.set_axis_off()
plt.tight_layout(); plt.savefig(FIG/'map_crawl_rate.png', dpi=130); plt.show()
top = z[z.n >= 500].join(shp.set_index('LocationID')[['zone', 'borough']] if 'zone' in shp else shp.set_index('LocationID')[['borough']]).sort_values('rate', ascending=False)
top.head(10).round(3)'''),
("md", "**Reading the map.** The darkest zones (about 30-38% crawl) are in Midtown and the east side of Manhattan; the rest of Manhattan is mostly orange-red (15-30%), some Brooklyn zones are moderate (about 10-20%), and Queens, the Bronx and the airports are nearly white. Crawling is a Manhattan-core phenomenon, not a citywide one. Grey zones have fewer than 100 May pickups."),
("md", "**Four-step rhythm for this chart (team: write steps 1 and 3 yourselves; step 1 *before* you rerun the cell)**\n1. What we expected: ______\n2. The chart (above)\n3. What it shows, in one sentence: ______\n4. Feature it suggests: zone crawl rates `pu_rate`, `do_rate`, and the borough flags"),
("md", "## 5. Which trips crawl?"),
("code", '''t = train.copy()
t['route'] = np.where(t.pickup_borough != t.dropoff_borough, 'across boroughs', np.where(t.pickup_borough == 'Manhattan', 'within Manhattan', 'within other borough'))
t['airport'] = np.select([t.PULocationID.isin(AIRPORTS), t.DOLocationID.isin(AIRPORTS)], ['airport pickup', 'airport drop-off'], 'no airport')
t['dist_band'] = pd.cut(t.trip_distance, [0, .5, 1, 2, 3, 5, 10, 200], labels=['<0.5', '0.5-1', '1-2', '2-3', '3-5', '5-10', '>10'])
out = {}
for col in ['route', 'airport', 'dist_band']:
    g = t.groupby(col, observed=True).crawl.agg(['size', 'mean']); g['share_of_all_crawls'] = t.groupby(col, observed=True).crawl.sum() / t.crawl.sum()
    out[col] = g.round(3); print(out[col], '\\n')
fig, ax = plt.subplots(figsize=(6, 3.5)); out['dist_band']['mean'].plot(kind='bar', ax=ax, color='#c0392b'); ax.set_ylabel('crawl rate'); ax.set_xlabel('measured trip distance (miles)'); plt.tight_layout(); plt.savefig(FIG/'rate_by_distance.png', dpi=130); plt.show()'''),
("md", "**Answer.** Crawling trips are overwhelmingly **short (under ~3 miles), within Manhattan**. Trips across boroughs and to or from airports almost never crawl. The measured distance here is only for describing the data; as a feature we may use only the planned distance."),
("md", "**Four-step rhythm for this chart (team: write steps 1 and 3 yourselves; step 1 *before* you rerun the cell)**\n1. What we expected: ______\n2. The chart (above)\n3. What it shows, in one sentence: ______\n4. Feature it suggests: planned distance `est_dist`; `same_borough`, `same_zone` and the airport flags"),
("md", "## 6. Is the positive rate stable from May to June to July?\nThis uses only the label rate, no model, so July stays unseen for modelling decisions."),
("code", '''df['hour'] = df.tpep_pickup_datetime.dt.hour
print(df.groupby('month').crawl.mean().round(4))
fig, ax = plt.subplots(figsize=(8, 3.8))
for mth, g in df.groupby('month'): g.groupby('hour').crawl.mean().plot(ax=ax, label={5: 'May', 6: 'June', 7: 'July'}[mth])
ax.legend(); ax.set_title('Crawl rate by hour, per month'); ax.set_ylabel('crawl rate'); plt.tight_layout(); plt.savefig(FIG/'stability_months.png', dpi=130); plt.show()
print(df.groupby('month').trip_distance.median().round(2).rename('median distance'))'''),
("md", "**Answer.** Not perfectly stable: the rate falls from about 19.3% (May) to 18.5% (June) to 16.3% (July). The shape across the day is the same, but July is lower at midday. This is concept drift, so July will be a harder, honest test and the reliability check in notebook 04 matters."),
("md", "## 7. Random split versus time split (Week 1 baselines, May and June only)\nA random split answers 'does this pattern exist in my data?'. A time split answers 'will it still hold next month?'. We score the two Week 1 baselines both ways. July stays untouched, so the random split pools **May and June only**. The no-skill PR AUC equals the positive rate, which differs between periods, so we also report **PR AUC lift = PR AUC / positive rate** (1 = no better than guessing); top-10% lift is shown too."),
("code", """from sklearn.model_selection import train_test_split
def score_baselines(tr, te):
    M_, _ = build_matrix(tr, {"te": te}); y = te.crawl.values
    rule = BusyRule().fit(tr); t2 = make_tree(2).fit(M_["train"][ALL_FEATS], tr.crawl.values)
    out = {}
    for name, p in [("busy-hours rule", rule.predict_proba(te)[:, 1]), ("depth-2 tree", t2.predict_proba(M_["te"][ALL_FEATS])[:, 1])]:
        m = metrics(y, p, .3); out[name] = {"positive rate": m["base_rate"], "PR AUC": m["pr_auc"], "PR AUC lift": m["pr_auc_lift"], "top-10% lift": m["lift@10%"]}
    return pd.DataFrame(out).T
pool = df[df.month.isin([5, 6])].reset_index(drop=True)
tr_r, te_r = train_test_split(pool, test_size=(pool.month == 6).mean(), random_state=42)
tr_r, te_r = tr_r.reset_index(drop=True), te_r.reset_index(drop=True)
cmp_split = pd.concat({"random split (May+June pooled)": score_baselines(tr_r, te_r), "time split (train May, test June)": score_baselines(train, val)})
cmp_split.round(3)"""),
("md", "**What it shows.** The two blocks are very close (depth-2 tree: PR AUC lift 2.34 random vs 2.41 time). We expected the random split to flatter the baselines, as it did in the lecture's example, but it does not here: May and June look alike, so a simple rule and a shallow tree do not memorise a period. The big shift only shows up in **July** (positive rate 16.3% vs 18.5%), which is why July is scored once at the end and compared by lift, not raw PR AUC. Only the time-split numbers are used for any decision."),
("md", "## 8. What the charts told us: features suggested\n| Chart | What it shows | Feature suggested |\n|---|---|---|\n| Hour x weekday heatmap | Crawls cluster in weekday daytime and the afternoon rush; nights are almost free | `hour`, `dow`, `is_weekend`, `is_rush` |\n| Zone map | Dark in Midtown and the east side of Manhattan, pale at the airports and in Queens | `pu_rate`, `do_rate`, borough flags |\n| Distance bands | Short trips crawl far more often than long ones | planned distance `est_dist` (estimated from May) |\n| Within or across boroughs, airports | Crawls are almost all within Manhattan; airport trips almost never crawl | `same_borough`, `same_zone`, airport flags |\n| Month-to-month rate | 19.3%, 18.5%, 16.3%: the rate drifts | none as a feature; a warning to re-check the threshold and to compare by lift, not raw PR AUC |"),
])

# ------------------------------------------------------------------ 03
n3 = nb([
("md", "# 03 - Features and pipeline (judged on June)\n\nEvery learned quantity (planned-distance proxy, zone crawl rates) is fitted on **May only** and applied unchanged to June. Inside May the quantities are computed **out-of-fold** so a training trip never sees its own label or distance."),
("code", SETUP),
("md", "## 1. A pipeline that cannot leak: three tests"),
("code", '''M, zh = build_matrix(train, {"val": val})
cols_used = set(ALL_FEATS) | {"PULocationID", "DOLocationID"}
# Test 1: no forbidden column is a model feature
assert not (set(ALL_FEATS) & set(FORBIDDEN)), "forbidden column in features"
# Test 2: transform ignores the target and distance of the data it transforms
v2 = val[["PULocationID", "DOLocationID", "trip_distance"]].copy(); v2["trip_distance"] = 999.0
assert zh.transform(v2).equals(zh.transform(val[["PULocationID", "DOLocationID", "trip_distance"]])), "transform reads trip_distance"
# Test 3: everything learned comes from May: tables only contain May's zone pairs
assert zh.tables_["pair"]["n"].sum() == len(train)
print("all three leak tests passed")
print("train/val feature shapes:", M["train"][ALL_FEATS].shape, M["val"][ALL_FEATS].shape)'''),
("md", "## 2. What each addition did on June\nLogistic regression and a decision tree (depth 8, min 1,000 per leaf), PR AUC on June. Base rate is the PR AUC of a useless model."),
("code", '''ytr, yva = train.crawl.values, val.crawl.values
sets = {"time only": TIME_FEATS, "+ geography flags": TIME_FEATS + GEO_FEATS, "+ planned distance": TIME_FEATS + GEO_FEATS + DIST_FEATS,
        "+ zone crawl rates": TIME_FEATS + GEO_FEATS + DIST_FEATS + ["pu_rate", "do_rate"], "+ zone-pair rate": ALL_FEATS}
rows = []
for name, c in sets.items():
    lr = make_logreg(c).fit(M["train"][c], ytr); tr_ = make_tree(8, 1000).fit(M["train"][c], ytr)
    rows.append({"feature set": name, "n_features": len(c),
                 "LogReg PR AUC": metrics(yva, lr.predict_proba(M["val"][c])[:, 1], .3)["pr_auc"],
                 "Tree PR AUC": metrics(yva, tr_.predict_proba(M["val"][c])[:, 1], .3)["pr_auc"]})
ladder = pd.DataFrame(rows).set_index("feature set").round(3); print(f"June base rate (PR AUC of a useless model): {yva.mean():.3f}"); ladder'''),
("md", "## 3. Features that lost, and the upper bound\nTwo reported on purpose: (a) `is_flex`, which adds nothing useful and whose payment type is only settled at the end; (b) the measured `trip_distance`, an **oracle that is not allowed at pickup**. Compare the oracle tree with the tree row above it: the real measured distance adds almost nothing over our zone-pair estimate, so the proxy captures what the planned distance would give."),
("code", '''c = ALL_FEATS
Xtr, Xva = M["train"][c].copy(), M["val"][c].copy()
res = {}
res["chosen features (LogReg)"] = metrics(yva, make_logreg(c).fit(Xtr, ytr).predict_proba(Xva)[:, 1], .3)["pr_auc"]
res["chosen features (tree depth 8, leaf 1000)"] = metrics(yva, make_tree(8, 1000).fit(Xtr, ytr).predict_proba(Xva)[:, 1], .3)["pr_auc"]
Xtr_f, Xva_f = Xtr.assign(is_flex=train.is_flex.values), Xva.assign(is_flex=val.is_flex.values)
res["+ is_flex (rejected: payment type is settled at the end)"] = metrics(yva, make_logreg(c + ["is_flex"]).fit(Xtr_f, ytr).predict_proba(Xva_f)[:, 1], .3)["pr_auc"]
Xtr_o, Xva_o = Xtr.assign(measured_dist=np.log1p(train.trip_distance.values)), Xva.assign(measured_dist=np.log1p(val.trip_distance.values))
res["ORACLE tree (depth 8, leaf 1000) + measured trip_distance (forbidden)"] = metrics(yva, make_tree(8, 1000).fit(Xtr_o, ytr).predict_proba(Xva_o)[:, 1], .3)["pr_auc"]
pd.Series(res).round(3).to_frame("June PR AUC")'''),
("md", "## 4. Tuning the tree on June\nOnly depth and minimum leaf size are tuned, only on June."),
("code", '''g = []
for d in (2, 3, 4, 6, 8, 10, 12):
    for leaf in (200, 1000, 3000):
        p = make_tree(d, leaf).fit(Xtr, ytr).predict_proba(Xva)[:, 1]; g.append({"depth": d, "min_leaf": leaf, "PR AUC": metrics(yva, p, .3)["pr_auc"]})
g = pd.DataFrame(g).pivot(index="depth", columns="min_leaf", values="PR AUC").round(3); g'''),
("code", '''best = g.stack().idxmax(); print("best tree (depth, min_leaf):", best, "PR AUC", round(g.stack().max(), 3))
Cs = {C: metrics(yva, make_logreg(ALL_FEATS, C).fit(Xtr, ytr).predict_proba(Xva)[:, 1], .3)["pr_auc"] for C in (0.01, 0.1, 1, 10)}
print("LogReg C vs June PR AUC:", {k: round(v, 4) for k, v in Cs.items()})'''),
("md", "**Conclusion.** The biggest gains come from the planned-distance proxy and the learned zone and zone-pair crawl rates; time and geography flags alone are weak. Notebook 04 compares the tuned models and finishes the evaluation."),
])

# ------------------------------------------------------------------ 04
n4 = nb([
("md", "# 04 - Evaluate, decide, and one look at July"),
("code", SETUP),
("code", '''M, zh = build_matrix(train, {"val": val, "test": test})
ytr, yva, yte = train.crawl.values, val.crawl.values, test.crawl.values
Xtr, Xva, Xte = M["train"][ALL_FEATS], M["val"][ALL_FEATS], M["test"][ALL_FEATS]
lr = make_logreg(ALL_FEATS).fit(Xtr, ytr)
# tree settings picked on June by PR AUC (same grid idea as notebook 03)
grid_ = {(d, l): metrics(yva, make_tree(d, l).fit(Xtr, ytr).predict_proba(Xva)[:, 1], .3)["pr_auc"] for d in (6, 8, 10, 12) for l in (200, 1000)}
best_dl = max(grid_, key=grid_.get); print("tree grid (depth, min_leaf) -> June PR AUC:", {k: round(v, 3) for k, v in grid_.items()}); print("picked:", best_dl)
dt = make_tree(*best_dl).fit(Xtr, ytr)'''),
("md", "## 1. The decision: what do errors cost?\nWe never warn for free. We state the costs, in 'units of one needless warning':\n\n* **Needless warning (false positive) = 1.** The passenger is told 'this may be slow', but it wasn't. Cost: a bit of annoyance, and a few passengers may cancel or doubt the next warning.\n* **Missed crawl (false negative) = 3.** The passenger gets stuck in traffic and a much longer, pricier ride than expected, with no heads-up. Worse for the customer and for trust in the service.\n\nWith well-calibrated probabilities the cost-minimising threshold is `c_FP / (c_FP + c_FN) = 1/4 = 0.25`. We pick the threshold that minimises cost on **June**, and show how sensitive it is to the 3:1 assumption."),
("code", '''pl, pd_ = lr.predict_proba(Xva)[:, 1], dt.predict_proba(Xva)[:, 1]
rows = []
for r in (1, 2, 3, 5, 10):
    tl, cl = best_threshold(yva, pl, r); td, cd = best_threshold(yva, pd_, r)
    rows.append({"cost ratio FN:FP": f"{r}:1", "LogReg thr": tl, "LogReg cost/1000": cl, "Tree thr": td, "Tree cost/1000": cd,
                 "cost if never warn": 1000 * r * yva.mean()})
pd.DataFrame(rows).round(3)'''),
("code", '''C_FN = 3
thr_lr, cost_lr = best_threshold(yva, pl, C_FN); thr_dt, cost_dt = best_threshold(yva, pd_, C_FN)
print("LogReg threshold", thr_lr, "cost/1000", round(cost_lr, 1), "| Tree threshold", thr_dt, "cost/1000", round(cost_dt, 1))
grid = np.linspace(.02, .9, 89)
plt.figure(figsize=(7, 3.8)); plt.plot(grid, [cost_per_1000(yva, pl, t, C_FN) for t in grid], label='LogReg'); plt.plot(grid, [cost_per_1000(yva, pd_, t, C_FN) for t in grid], label='Tree')
plt.axhline(1000 * C_FN * yva.mean(), color='grey', ls='--', label='never warn'); plt.xlabel('warning threshold'); plt.ylabel('cost per 1,000 trips (June)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'cost_curve.png', dpi=130); plt.show()'''),
("md", "## 2. Model comparison on June\nBaselines use the same cost view. Precision, recall, PR AUC, lift, and the share of trips warned. Two lifts are shown: `lift@10%` (crawl rate in the top-scored 10% divided by the overall rate) and `pr_auc_lift` (PR AUC divided by the positive rate), which is the fair way to compare months because the no-skill PR AUC moves with the positive rate."),
("code", '''rule = BusyRule().fit(train); b2 = rule.predict_proba(val)[:, 1]
tree2 = make_tree(2).fit(Xtr, ytr); b3 = tree2.predict_proba(Xva)[:, 1]
thr_b3, _ = best_threshold(yva, b3, C_FN)
tab = {"1 never warn": metrics(yva, np.zeros(len(yva)), .5), "2 busy-hours rule": metrics(yva, b2, .5), "3 depth-2 tree": metrics(yva, b3, thr_b3),
       "Logistic regression": metrics(yva, pl, thr_lr), "Decision tree": metrics(yva, pd_, thr_dt)}
cost_tab = {"1 never warn": cost_per_1000(yva, np.zeros(len(yva)), .5, C_FN), "2 busy-hours rule": cost_per_1000(yva, b2, .5, C_FN), "3 depth-2 tree": cost_per_1000(yva, b3, thr_b3, C_FN),
            "Logistic regression": cost_lr, "Decision tree": cost_dt}
cmp_ = pd.DataFrame(tab).T; cmp_["cost/1000 (3:1)"] = pd.Series(cost_tab); cmp_.round(3)'''),
("md", "## 3. Reliability check (calibration, June)\nIf a model says 70%, do about 70% of those trips crawl? This matters because the cost threshold assumes calibrated probabilities."),
("code", '''from sklearn.calibration import calibration_curve
plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], 'k--', label='perfect')
for name, p in [("LogReg", pl), ("Tree", pd_)]:
    fr, mp = calibration_curve(yva, p, n_bins=10, strategy="quantile"); plt.plot(mp, fr, 'o-', label=name)
plt.xlabel('predicted probability'); plt.ylabel('observed crawl rate'); plt.legend(); plt.title('Reliability (June)'); plt.tight_layout(); plt.savefig(FIG/'calibration.png', dpi=130); plt.show()
from sklearn.metrics import brier_score_loss
print("Brier score  LogReg", round(brier_score_loss(yva, pl), 4), " Tree", round(brier_score_loss(yva, pd_), 4), " (never-warn constant:", round(brier_score_loss(yva, np.full(len(yva), ytr.mean())), 4), ")")'''),
("md", "## 4. Choose the model\nThe model with the lower June cost is chosen (PR AUC is reported next to it). The decision is made here, **before** July is touched."),
("code", '''chosen = "LogReg" if cost_lr <= cost_dt else "Tree"
model, thr = (lr, thr_lr) if chosen == "LogReg" else (dt, thr_dt)
print("CHOSEN:", chosen, "threshold", thr)'''),
("md", "## 5. Results by slice (June)\nWhere does the warning work, and which trips does it miss?"),
("code", '''v = val.copy(); v["p"] = model.predict_proba(Xva)[:, 1]; v["warn"] = (v.p >= thr).astype(int)
v["hour_band"] = pd.cut(v.tpep_pickup_datetime.dt.hour, [-1, 5, 9, 15, 19, 23], labels=["night 0-5", "am rush 6-9", "midday 10-15", "pm rush 16-19", "evening 20-23"])
v["route"] = np.where(v.pickup_borough != v.dropoff_borough, "across boroughs", np.where(v.pickup_borough == "Manhattan", "within Manhattan", "within other borough"))
v["airport"] = np.where(v.PULocationID.isin(AIRPORTS) | v.DOLocationID.isin(AIRPORTS), "airport", "no airport")
v["weekend"] = np.where(v.tpep_pickup_datetime.dt.dayofweek >= 5, "weekend", "weekday")
v["dist_band"] = pd.cut(v.trip_distance, [0, 1, 2, 3, 5, 200], labels=["<1", "1-2", "2-3", "3-5", ">5"])
v["flex"] = np.where(v.is_flex == 1, "flex fare", "metered")
def slice_table(col):
    out = []
    for k, g in v.groupby(col, observed=True):
        tp = ((g.warn == 1) & (g.crawl == 1)).sum()
        out.append({col: k, "trips": len(g), "crawl_rate": g.crawl.mean(), "precision": tp / max(g.warn.sum(), 1), "recall": tp / max(g.crawl.sum(), 1), "warned_%": 100 * g.warn.mean(), "missed_crawls": int(((g.warn == 0) & (g.crawl == 1)).sum())})
    return pd.DataFrame(out).set_index(col).round(3)
for col in ["hour_band", "weekend", "route", "airport", "dist_band", "flex"]:
    print(slice_table(col), "\\n")'''),
("code", '''miss = v[(v.warn == 0) & (v.crawl == 1)]
print(f"Missed crawls: {len(miss):,} of {v.crawl.sum():,} ({len(miss)/v.crawl.sum():.1%})")
print(miss.groupby("hour_band", observed=True).size().div(len(miss)).round(3).rename("share of missed crawls by hour band"))
print(miss.groupby("route").size().div(len(miss)).round(3).rename("share of missed crawls by route"))
print(miss.pickup_zone.value_counts().head(8).rename("zones with most missed crawls"))'''),
("md", "## 6. July: scored once\nThe model, features, thresholds and tables are frozen. This is the only cell that evaluates July. The bootstrap resamples trips 500 times for 95% intervals."),
("code", '''pte = model.predict_proba(Xte)[:, 1]
july = metrics(yte, pte, thr); july["cost_per_1000"] = cost_per_1000(yte, pte, thr, C_FN)
ci = bootstrap(yte, pte, thr, C_FN, n_boot=500)
res = pd.DataFrame({"July": pd.Series(july)}).join(ci)
res["June (validation)"] = pd.Series({**metrics(yva, model.predict_proba(Xva)[:, 1], thr), "cost_per_1000": cost_per_1000(yva, model.predict_proba(Xva)[:, 1], thr, C_FN)})
never = cost_per_1000(yte, np.zeros(len(yte)), .5, C_FN)
print(f"July base rate {yte.mean():.3%}; cost if we never warn: {never:.1f} per 1000; with the model: {july['cost_per_1000']:.1f}")
res.round(3)'''),
("md", "Interpretation of the July numbers is in `MODEL_CARD.md`. The tail of the cell prints the table once; no further tuning happens after it."),
])

for name, n in [("01-frame", n1), ("02-clean-and-split", n2), ("03-features", n3), ("04-evaluate", n4)]:
    nbf.write(n, Path(__file__).parent / f"{name}.ipynb")
print("notebooks written")
