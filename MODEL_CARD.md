# MODEL_CARD.md - Crawl warning model

## What it does
For a yellow-taxi trip about to start, outputs the probability that its average speed will be under 6 mph. A dispatcher warns the passenger when the probability is at or above **0.31**.

## Model
Decision tree (depth 12, minimum 200 trips per leaf, depth and leaf size chosen on June by PR AUC), 17 features: hour, weekday, weekend and weekday-rush flags, same zone / same borough, airport pickup / drop-off, Manhattan / Queens / Brooklyn flags, estimated planned distance (zone-pair median of earlier trips), and smoothed crawl rates of the pickup zone, drop-off zone and zone pair. A logistic regression with the same features was close (June PR AUC 0.619 vs 0.628). All learned tables use May only.

## Data
600,000 yellow-taxi trips (May-July 2026, 200,000 per month). Training: May. Validation and every choice: June. One final test: July. After cleaning (notebook 02): 559,203 trips (93.2%), of which 187,612 train, 186,114 validation, 185,477 test.

## The decision: costs and threshold
* Needless warning = **1** unit (annoyed or doubtful passenger, possible cancellation).
* Missed crawl = **3** units (passenger stuck in a slow, longer and costlier ride with no warning, and loses trust in the service).
* With calibrated probabilities the best threshold is 1/(1+3) = 0.25. Minimising June cost gives **0.31** (on June the tree slightly over-predicts in the middle and upper score range, e.g. trips scored about 0.47 crawled 43% of the time, which fits a threshold a little above 0.25; its Brier score on June is 0.102 versus 0.151 for a constant guess).
* Sensitivity (June): the threshold moves from about 0.5 at 1:1 to 0.33 at 2:1, 0.19 at 5:1 and 0.10 at 10:1. The cost advantage over never warning holds at every ratio.
* These costs are our argued assumptions, not measured business numbers. A real operator should replace them.

## Results
| | PR AUC | PR AUC lift | Precision | Recall | Lift@10% | Cost/1,000 |
|---|---|---|---|---|---|---|
| June (validation) | 0.628 | 3.39 | 0.521 | 0.697 | 3.81 | 287 |
| **July, scored once** | **0.610** [0.604, 0.615] | **3.74** [3.69, 3.78] | **0.471** [0.467, 0.476] | **0.745** [0.741, 0.750] | **4.11** [4.07, 4.16] | **261** [258, 264] |
| July, never warn | 0.163 (base rate) | 1.00 | 0 | 0 | 1 | 490 |

(95% bootstrap intervals, 500 resamples of July trips.) The model roughly halves the expected cost versus never warning, and beats all three baselines on June. The intervals only reflect sampling noise inside July; they do not include month-to-month drift.

## Stability
PR AUC lift (PR AUC divided by the positive rate) is the fair comparison across months, since the no-skill PR AUC equals the positive rate: 3.39 in June and 3.74 in July. So relative to guessing the model did slightly *better* in July, even though raw PR AUC fell from 0.628 to 0.610. The crawl rate falls from 19.3% (May) to 18.5% (June) to 16.3% (July) while the median trip distance stays about the same (1.9-2.0 miles), so trips really got faster in July. The model keeps its ranking quality (PR AUC 0.610 vs 0.628) and its recall rose (0.745 vs 0.697) but precision fell (0.471 vs 0.521) because fewer trips crawl: some warnings are now needless. A fixed threshold should be re-checked monthly.

## Where the warning works
Within Manhattan on weekdays between 10:00 and 19:00. Short routes: for trips under 1 mile (38% crawl rate) precision is 0.59 and recall 0.76.

## Which trips the warning misses (plainly)
On June, 10,447 of 34,502 crawls (30.3%) get no warning.
* **96% of the missed crawls are within Manhattan**, mostly in busy daytime zones (top zones: Upper East Side South and North, East Chelsea, Clinton East, East Village, Times Sq): trips that are slow but fall below the 0.31 threshold because their zone pair is only moderately slow.
* **Weekends:** recall is 0.47 versus 0.75 on weekdays. Weekend crawls are rarer (about 12% of trips versus 21%), so fewer weekend trips reach the threshold. We did not test further why.
* **Evenings 20-23h:** recall 0.45; **nights 0-5h:** recall 0.11 (few crawls, 3.6% of trips).
* **Crawls outside Manhattan or across boroughs** (rare, 0.8% and 3.3% crawl rates) are almost never caught (recall about 7%).
* **Airport trips:** crawl rate 0.3%; the model never warns (nothing to catch).
* **Flex-fare trips** (about 25% of cleaned trips): recall 0.51 versus 0.73 for metered trips. Flex trips crawl half as often (10% vs 21%) and are more weekend-heavy (37-42% vs 24-29%), which overlaps with the weekend weakness. We did not test the cause.
* **Precision:** about half of the warnings are needless (precision 0.47 on July).

## Limits and risks
* The planned distance is a proxy built from past measured trips on the same zone pair; a real routing distance would be better and may change results. Zone-level resolution hides block-level traffic.
* No weather, events, or live traffic: a rainy rush hour or a parade is invisible to the model.
* Cleaning removed 6.8% of trips (zero distances, impossible durations, etc.). Real unlabelable trips would get no prediction in production.
* About 30% of crawls are missed (June) and about half of warnings are needless; it should be used as a soft heads-up, not a promise.
* Drift: July positive rate is 3 points lower than May.
