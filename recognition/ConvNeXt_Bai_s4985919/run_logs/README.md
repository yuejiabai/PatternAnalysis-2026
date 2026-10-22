# run_logs: original evidence for the numbers in the report

Everything in this folder was written automatically by the project scripts during my own runs on Rangpur (and, where stated, the EAIT GPU Lab) and is committed unchanged. It contains settings and aggregate metrics only: no MRI images, no model weights, and no scan or subject IDs.

| Folder / file | What it is | Produced by | Used in README section |
|---|---|---|---|
| `convnext/<run>/`, `cnn/<run>/` | One folder per training run (13 runs). `config.json` = all training settings; `history.json` = training loss and validation metrics after every epoch; `best_meta.json` = which epoch was kept and why, plus fingerprints (SHA-256 hashes) of the train/validation/test file lists, so a checkpoint can only be evaluated on exactly the split it was trained for. Early runs have no `best_meta.json` (written before that feature existed). | `train.py` | 9 (experiment history) |
| `summary.csv` | One line per run: settings, best validation accuracy and the epoch that was saved. | `collect_runs.py` | 9 |
| `evaluation/analysis_convnext_ensemble.json`, `evaluation/analysis_cnn.json` | Every validation and test number for the final models: accuracy, AUROC, confusion matrix, calibration (ECE, Brier, reliability bins), reject rule, subgroup accuracy, failure cases, parameter count, latency and GPU memory. | `analyze.py` | 2, 5, 6, 7, 8 |
| `evaluation/<project>__<output folder>__<split>_metrics.json` | Metric files of earlier evaluations (slice, scan and patient level), e.g. the strict-split test run of run 3. | `predict.py` | 2, 9 |
| `slurm/slurm_training_jobs.txt` | Slurm accounting (`sacct`) for the training jobs: job ID, state and elapsed time. | Rangpur `sacct` | 8 (training time) |
| `slurm/slurm_job_to_run.txt` | First line of each training job's log: which run (input mode, seed, run name) each job ID trained. | `run_recipe.slurm` | 8 |
| `slurm/slurm_job_history.txt` | Full list of my Rangpur jobs since 1 Oct 2026, including interactive sessions and failed or cancelled attempts. | Rangpur `sacct` | – (provenance) |
| `diagnostics/development/` | Development audit of the fixed run-3 model on training and validation subjects only (no test data): `report.md` = readable summary; `summary.json` = metrics with patient-level bootstrap intervals; `check_summary.json` = data checks (subject overlap, slice numbering). | `diagnose_development.py` | 9 |
| `diagnostics/brightness/` | Brightness-sensitivity experiment on 280 validation slices (70 subjects): `report.md` = readable summary; `condition_metrics.csv` = metrics per brightness factor and class (empty cells: a metric is undefined when a class is absent); `summary.json` = the same with bootstrap intervals. | `test_brightness_sensitivity.py` | 9 |
| `adni_v2/PROTOCOL.md` | Evaluation protocol for the corrected subject-disjoint split, committed before any test evaluation on that split. | written by me | 2, 5 |
