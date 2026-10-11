# BaseballIQ — pitcher analysis

Pitch-tracking data turned into audited pitcher reports, a whiff model, and coaching hypotheses a coach can test.

**Research question.** Among ordinary (non-bunt) swings, how well do pitch characteristics and context predict a miss?
The model is retrospective: it uses the measured location of a pitch that has already been thrown.

## What the project does

1. **Ingest and audit.** Downloads MLB Statcast one day at a time, reconciles it against the schedule, and checksums
   every file. A public college TrackMan dataset is ingested the same way. Both are mapped to one canonical schema
   with explicit sign and unit conventions, and every row is kept with a status rather than silently dropped.
2. **Build the modelling dataset.** Defines the model population, derives the swing table, and splits it
   chronologically. Labels come from one mapping file; an unmapped outcome stops the pipeline.
3. **Lock the evaluation before fitting.** The protocol, feature allowlist and splits are hashed before any model is
   fitted, and the 2026 season is stored sealed. A test fails if a locked file changes.
4. **Fit and select a whiff model.** Transparent baselines (constant, pitch type, logistic linear and spline) are
   compared with tree ensembles on validation data. Calibration is decided by a rule written down in advance.
5. **Test once on the sealed season.** The frozen model is scored a single time on 2026; the script refuses a second
   scoring.
6. **Diagnose and revise.** The 2026 strike-zone definition changed, so a second model uses a location feature that
   means the same thing in every season. It is frozen and will be tested on the 2027 season, behind a drift check.
7. **Compare pitchers.** Each pitcher's observed whiff rate is compared with the rate expected from a model that never
   saw that pitcher, adjusted for the batters he faced. Flags carry the evidence behind them.
8. **Check transfer to college data.** Reduced-feature MLB models are scored on college TrackMan under a protocol
   locked before scoring.
9. **Generate coaching hypotheses.** Swing, whiff and run-value models propose changes only inside what a pitcher has
   already shown he can throw. Model-only evidence is capped at the lowest tier; higher tiers need the pitcher's own
   observed outcomes to agree.

## Results

| Model | Result |
|---|---|
| Whiff model, sealed 2026 test | ROC-AUC 0.787; beats every baseline; does not meet the calibration criteria |
| Whiff model v2 | Frozen; partial calibration fix confirmed on development data; clean test is the 2027 season |
| Swing model, August 2026 | 78.8% accuracy, ROC-AUC 0.876 (always guessing the more common outcome: 52.4%) |
| Swing model, September 2026 (not used to fit or choose) | 78.6% accuracy, ROC-AUC 0.875 |
| College transfer | Ranking holds for this program's scrimmages only; calibration is weak |

The swing model uses each batter's own swing tendencies, computed only from his earlier games.

## Setup and reproduction

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python src/run_pipeline.py              # one-week sample + college data; downloads once, then reuses checksummed files
.venv/bin/python src/run_pipeline.py --season     # + model dataset and baselines on the full 2025 regular season
.venv/bin/python src/run_pipeline.py --comparison # + 2026 processing, drift checks and pitcher comparison (~500 MB downloaded)
.venv/bin/python -m pytest -q tests               # also run by run_pipeline.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_pitcher_data_exploration.ipynb
```

Only code is committed. The pipeline writes data to `data/`, fitted models to `models/` and reports to `reports/`.
`run_pipeline.py` also writes `reports/run_record.json` with package versions, seed, config and manifest hashes, and
output checksums. Tests that need a pipeline output skip until it exists.

| Step | Script | Output |
|---|---|---|
| Statcast download (one request per day, schedule reconciliation) | `src/ingest.py` | `data/raw/statcast/`, `data/manifests/` |
| College TrackMan (checksum vs Mendeley) | `src/ingest_trackman.py` | `data/raw/trackman_college/`, manifest |
| Canonical contract, labels, row status | `src/normalize.py`, `src/normalize_trackman.py` | `data/processed/*.parquet`, exclusion logs |
| Audits | `src/validate_data.py`, `src/validate_trackman.py` | `reports/data_audit.md`, `reports/trackman_audit.md`, `reports/data_dictionary.csv` |
| Descriptive analysis | `src/pitcher_analysis.py [--pitcher-id ID]` | `reports/pitcher_analysis.md`, `reports/pitcher_summary.csv`, `reports/figures/` |
| Population, swing table, splits, label checks | `src/build_dataset.py --source statcast_2025` | `data/model/<source_id>/`, `data/splits/<source_id>.json`, `reports/dataset_<source_id>.md` |
| Baselines and pooled logistic models | `src/train_baselines.py` | `models/baselines_v1/`, `reports/baseline_report.md` |
| Calibration decision | `src/calibrate.py` | `models/calibration_v1/decision.json`, `reports/calibration_report.md` |
| Tree-model selection and learning curve | `src/model_selection.py` | `models/selection_v1/`, `reports/model_selection_report.md` |
| Final fit (frozen) | `src/final_fit.py` | `models/final_v1/` |
| One-time test (dry run first) | `src/final_test.py [--dry-run]` | `reports/final_test.md`; refuses a second scoring |
| Label-free drift check (gate before any sealed test) | `src/drift_check.py --features v1\|v2` | `reports/drift_*.md` |
| Model v2 (one-time freeze) | `src/model_v2.py` | `models/final_v2/`, `reports/model_v2_report.md` |
| v2 test on 2027 (dry run on 2026 first) | `src/final_test_v2.py [--dry-run]` | `reports/final_test_v2*.md` |
| 2025 vs 2026 batter behaviour | `src/season_shift.py` | `reports/season_shift_2025_2026.md` |
| Expected whiff (pitcher-disjoint, batter-adjusted) | `src/expected_whiff.py` | `data/pitcher_comparison/` |
| Pitcher comparisons and player reports | `src/player_analysis.py [--pitcher-id ID]` | `reports/pitcher_comparison*.{md,csv}`, `reports/pitcher_flags.csv`, `reports/players/` |
| College transfer (protocol locked first) | `src/transfer.py` | `models/transfer_reduced_v1/`, `reports/trackman_transfer_report.md` |
| Outcome models (swing, run value; whiff reused from v2) | `src/outcome_models.py` | `models/outcome_models_v2/`, `data/outcomes/pitches_all.parquet`, `reports/outcome_models.md` |
| Hypotheses (rules in `config/hypotheses.yaml`) | `src/hypotheses.py [--pitcher-id ID]` | `reports/hypotheses.csv`, `reports/hypotheses_report.md`, `reports/players/<id>_hypotheses.md` |
| Sealed 2026 snapshot (once) | `src/ingest.py --source statcast_2026_sealed` | `data/sealed/` (not read by development code) |

## Data sources

- **MLB Statcast**, Baseball Savant CSV endpoint, 2025 and 2026 regular seasons.
  Data © MLB Advanced Media; research use. Review MLB terms before distributing any product built on it.
- **College TrackMan**: Pifer, N. D. (2024). *Optical Tracking Data from College Baseball Scrimmages* (v3).
  Mendeley Data. https://doi.org/10.17632/xfnz6mkdzm.3. **CC BY 4.0**; this project changes the data
  (sign conventions, unit conversions, reconstructed game IDs), as documented in `config/field_mapping.yaml`.

## Rules worth knowing

- **Labels** come only from `config/outcome_mapping.yaml`. An unmapped outcome value stops the pipeline rather than being guessed.
- **Leakage**: `miss_distance` (present only on misses), bat-tracking fields and all batted-ball/result columns are
  denylisted in `config/field_mapping.yaml`; models use an allowlist (`config/features.yaml`, `config/features_v2.yaml`).
- **Conventions**: all horizontal values are catcher's view, +x toward 1B. `*_arm` columns are arm-side positive for
  both hands. TrackMan's horizontal fields are pitcher's view and are negated on import. TrackMan `pfxx`/`pfxz` are
  *not* Statcast `pfx_x`/`pfx_z`.
- **Pooled model first**: one model across pitchers; per-pitcher reports come from it, not from per-pitcher models.
- **Every pitch is kept** (takes included) in `model_pitches.parquet`; the whiff model uses the derived `swings.parquet`.
- **Position players pitching** are identified by MLB primary position (not P/TWP) and excluded from the model population only.
- **College pitch types**: operator tags are primary, TrackMan auto labels are kept, and findings are shown under both.
- **Never pool** Statcast and TrackMan rows into one training table. The college file is one program's mostly
  intrasquad scrimmages: use it descriptively, and treat any transfer evaluation as experimental.
- **The 2026 test has been used.** Any revised model is evaluated on the 2027 regular season. Known issue: 2026
  `sz_top`/`sz_bot` follow the ABS height-based zone, so `plate_z_zone` means something different from 2025.
- **Hypotheses are prompts for a coach-run experiment, never prescriptions.** The models are associational.
- Row statuses: `retained` / `excluded` (rule) / `quarantined` (untrustworthy key). Implausible values are flagged in `review_flags`, never deleted.
