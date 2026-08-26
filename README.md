# SDSS17: Supervised vs Unsupervised Classification of Celestial Objects

**English** · [Русский](README.ru.md)

Comparison of Random Forest (supervised) and K-Means (unsupervised) on a single task:
separating STAR / GALAXY / QSO by photometry and redshift.

**Main deliverable — [`REPORT.md`](REPORT.md):** a consolidated report with every table,
number and figure. It is assembled automatically from the saved CSVs — no value in it is
typed by hand. (The report itself is written in Russian.)

## Quick start

```bash
uv sync
./run_all.sh
```

A full run takes ~10 minutes, dominated by `GridSearchCV` and the control experiments.
Every output file is overwritten; no manual edits are needed between stages.

## Data

`data/raw_sdss17.csv` is the original `star_classification.csv` from Kaggle
([Stellar Classification Dataset – SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)),
100,000 objects. The file is not committed (see `.gitignore`) — download it and drop it there.

If the file is missing, `scripts/00_fetch_data.py` pulls an equivalent sample straight from
the primary source: the public SDSS SkyServer DR17 SQL service (no registration), using the
same `PhotoObj`/`SpecObj` join and the same column schema. Only the particular slice of rows
differs; the resulting class balance is nearly identical. If the file is already in place,
the script does nothing.

## Layout

```
data/       raw_sdss17.csv (source), processed.csv (cleaned, with split and spec_obj_id columns)
models/     scaler.pkl, random_forest_model.pkl, kmeans_model.pkl — not committed
results/    23 tables: metrics, predictions, statistics, control experiments
figures/    15 plots, PNG at 300 dpi
scripts/    00..09 pipeline stages; config.py holds shared constants, meta.py passes data between stages
```

## Pipeline stages

| Script | What it does |
|---|---|
| `00_fetch_data.py` | obtain `data/raw_sdss17.csv` |
| `01_preprocess.py` | cleaning, feature selection, `StandardScaler`, stratified 80/20 split |
| `02_train_supervised.py` | Approach A: `GridSearchCV` (5 folds) + Random Forest |
| `03_cluster_unsupervised.py` | Approach B: elbow/silhouette for k=2..8, K-Means k=3, PCA |
| `04_evaluate.py` | metrics, confusion matrix, cluster×class table, `run_config.json` |
| `05_figures.py` | the four required plots |
| `06_extended_analysis.py` | per-class metrics, ROC/AUC, statistics, correlations, colour indices |
| `07_control_experiments.py` | 7 feature sets + learning curve |
| `08_extended_figures.py` | 11 additional plots |
| `09_build_report.py` | assembles `REPORT.md` from all tables |

Stages 05–09 read only the saved CSVs and never load the models, so figures and the report
can be rebuilt without retraining.

## Key decisions

- **Features:** `u, g, r, i, z, redshift`. Coordinates and technical IDs are deliberately
  excluded — they inflate accuracy without any physical meaning (see control experiments).
- **Identifiers are read as strings.** SDSS `obj_ID` and `spec_obj_ID` are on the order of
  1.2·10^18, above 2^53, and some values in the CSV are written in scientific notation. With
  type inference pandas picks `float64` and irreversibly drops the low-order digits: distinct
  objects collapse onto one identifier, and `obj_id` in the output tables degrades into
  `1.2376609613304302e+18`.
- **Deduplication is keyed on `spec_obj_ID`, not `obj_ID`.** The unit of observation here is
  the spectrum — the class is derived from it, and `spec_obj_ID` is unique across all 100,000
  rows. `obj_ID` is not a key: it repeats across 19,154 rows whose photometry, class and
  redshift all differ. Deduplicating on it would discard ~19k genuine observations (22% of the
  dataset).
- **Cleaning:** rows are dropped when photometry falls outside 0..40 mag (SDSS encodes missing
  photometry as −9999) or `redshift` falls outside −0.1..10. On the Kaggle version exactly one
  row is affected, leaving 99,999 objects.
- **The scaler** is fitted on the training split only and then applied to the whole set. This
  keeps the Approach A evaluation leak-free; Approach B reuses the same scaler.
- **`silhouette_score`** is computed on a deterministic 10,000-object subsample — the full set
  would require a 100,000 × 100,000 distance matrix.
- **Reproducibility:** `random_state=42` is fixed for the split, Random Forest, K-Means
  (`n_init=10`), PCA and the silhouette subsample. Two consecutive full runs produce
  byte-identical tables; only the training-time measurements differ.

## Results

Kaggle SDSS17, 99,999 objects after cleaning (GALAXY 59.4%, QSO 19.0%, STAR 21.6%).

| Approach | Metric | Value |
|---|---|---|
| A (Random Forest) | accuracy | 0.9796 |
| A | precision / recall / f1 (macro) | 0.9802 / 0.9722 / 0.9761 |
| A | training time | 9.1 s (+ ~230 s for `GridSearchCV`) |
| B (K-Means, k=3) | ARI | 0.2580 |
| B | NMI | 0.3094 |
| B | purity | 0.7340 |
| B | silhouette | 0.4426 |
| B | training time | 0.38 s |

Best Approach A configuration: `n_estimators=300`, `max_depth=20` (cv f1_macro = 0.9747).

### What the control experiments showed

| Feature set | RF f1_macro | K-Means ARI |
|---|---|---|
| working set (baseline) | 0.9761 | 0.2580 |
| + coordinates | 0.9759 | 0.2611 |
| coordinates only | 0.5497 | −0.0023 |
| without redshift | 0.8373 | 0.0207 |
| redshift only | 0.9375 | 0.2525 |
| colour indices + redshift | 0.9760 | **0.3133** |

- **Coordinates leak.** On `alpha` and `delta` alone accuracy reaches 0.6865, against 0.5945
  for trivially predicting the majority class. There is no physics behind this — the model
  memorises the survey geometry. That is precisely why coordinates are excluded.
- **Redshift nearly solves the task on its own** — f1_macro 0.9375 from a single feature.
- **Switching to colour indices improves clustering**: ARI rises from 0.2580 to 0.3133 with
  the same algorithm. It changes nothing for Random Forest, which constructs feature
  differences by itself.

### Main observations

1. **Silhouette does not confirm k = 3** — it peaks at k = 2 (0.503 against 0.443), and
   inertia decays smoothly with no pronounced elbow. Without a hint from the labels, the
   geometry of the data does not yield three groups.
2. **No cluster corresponds to the STAR class.** Majority vote yields GALAXY, QSO, GALAXY:
   K-Means cuts the elongated galaxy cloud in half, and stars dissolve inside both galaxy
   clusters (11,248 and 10,343 out of 21,593). Quasars, by contrast, separate cleanly
   (purity 0.938) thanks to their high redshift.
3. **Approach A's weak spot is quasars**: recall 0.9285, confused with galaxies. Stars
   separate almost perfectly (recall 0.9995, ROC-AUC 0.9998).
4. **Data is more than sufficient**: the learning curve plateaus at 20,000–40,000 objects.

## Stack

Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib. Dependencies are
pinned in `uv.lock`; linting via ruff.
