# Will this trip crawl? (Team C, Project 1)

Predicts, at pickup, whether a New York yellow-taxi trip will average under 6 mph. See `PROBLEM.md` (question, target, baselines) and `MODEL_CARD.md` (results, costs, failures).

## Run everything from a fresh clone
```bash
git clone <repo-url> && cd <repo-folder>
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows (use .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python data/get_data.py              # downloads into data/raw/ (git-ignored)
jupyter nbconvert --to notebook --execute --inplace notebooks/01-frame.ipynb notebooks/02-clean-and-split.ipynb notebooks/03-features.ipynb notebooks/04-evaluate.ipynb
```
Or open the notebooks in order with `jupyter lab`. The whole run takes a few minutes. Figures are written to `reports/figures/`.

## Layout
| Path | What |
|---|---|
| `data/get_data.py` | download script (raw data is not committed) |
| `src/clean.py` | cleaning function `build_dataset()` and target definition, rebuilds everything from raw |
| `src/features.py` | leak-free features, `ZoneHistory` (learns only from May, out-of-fold on May), `FORBIDDEN` list |
| `src/models.py`, `src/evaluate.py` | models, baselines, metrics, costs, bootstrap |
| `notebooks/01..04` | frame, clean and split, features, evaluate (July scored once, only in 04) |
| `notebooks/_build.py` | script that generates the notebooks (optional) |

## Declaration of AI use
-----

## Quick guide to the main ideas (for the Defend hour)
* Split: May train, June validation and all choices, July one final look.
* Planned distance is allowed (known at booking); measured distance is not. We estimate the plan from the median distance of earlier trips on the same zone pair, learned from May only, and out-of-fold inside May.
* Target rate encodings (zone crawl rates) are also out-of-fold in May; otherwise a trip would see its own label.
* Costs: a missed crawl costs 3, a needless warning 1, so the threshold is below 0.5 (0.25 in theory, 0.31 from June).

## Collaboration test

Contribution by elisaisufi.