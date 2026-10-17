# ConvNeXt (from scratch) for AD vs NC classification on ADNI brain MRI

COMP3710 Pattern Recognition — Hard difficulty — Student s4985919

**Task.** Classify ADNI brain MRI slices as Alzheimer's disease (AD) or cognitively normal (NC) with a ConvNeXt-style network that is defined and trained from scratch (no pretrained weights, confirmed by course staff). The metric agreed with the teaching team is **slice-level accuracy on the official test folder**; the target in the spec is 0.80.

## 1. Headline result (read this first)

**The 0.80 target was not reached. The best test slice accuracy is 75.31 %.** That number comes with a data-leakage caveat, so two results are reported side by side:

| Setting | Model | Test slice accuracy | AUROC | What it measures |
|---|---|---|---|---|
| **Official split** (course folders as given) | ConvNeXt, 3-seed ensemble, 3-slice context input | **75.31 %** | 0.847 | Not subject-independent: 186 of 452 training subjects and 30 of 113 validation subjects also appear in the test folder (different visits) |
| **Strict split** (216 test subjects removed from training) | ConvNeXt, single model, 8 epochs, older recipe | **65.78 %** | 0.73 | Subject-independent generalisation |

Why both: the course folders are separated by *scan* (image ID), not by *subject*. 216 of the 331 test subjects also have another visit in the training folder. The PDF asks for strict patient-level separation, so the strict number is the honest estimate of generalisation to new patients. The official-split number is reported because it is what the course folders produce.

The same official-split ensemble, broken down by whether the test subject was seen in training:

| Test subjects | Slices | Accuracy |
|---|---|---|
| Seen in training folder (216 subjects) | 5,560 | **90.4 %** |
| Not seen (115 subjects) | 3,440 | **50.8 %** (chance level) |

So most of the official-split score comes from recognising subjects, not diagnosing disease. The model does not reach 80 % on unseen patients, and I did not find a legitimate way to do so (Section 8).

The official test folder had already been evaluated during development, before this write-up. Checkpoints (best epoch), the decision threshold (0.5, fixed before training) and the reject rule were selected on validation data only. No model, threshold or hyper-parameter was changed after looking at the final test breakdowns in Sections 5–7. **[TODO: state honestly how the context3 input and the 3-seed ensemble were chosen — on validation numbers, or after also seeing test accuracies of single runs.]**

## 2. Data and splits

- ADNI AD_NC slices, 20 slices per scan. A filename `scanID_sliceNo.jpeg` carries the ADNI *Image ID* of a scan, not a patient ID. The patient ID (e.g. `137_S_4672`) is read from the metadata JSON (`raw` path, regex `\d{3}_S_\d{4}`).
- Labels: AD = class 0, NC = class 1 (ImageFolder order).
- Test folder: 331 subjects, 450 scans, 9,000 slices (4,460 AD, 4,540 NC).
- Validation is always split **by subject** (seed 42, 20 %, stratified), in both modes, and train/validation never share a subject (asserted in code).

| Mode | Training | Validation | Note |
|---|---|---|---|
| `strict` | 279 subjects, 10,080 slices | 70 subjects, 2,620 slices | 8,820 slices (41 %) removed because their subject is in the test folder |
| `official` | 452 subjects, 17,380 slices | 113 subjects, 4,140 slices | Prints a leakage warning; 186 train + 30 val subjects overlap with test |

`python dataset.py strict` or `python dataset.py official` prints these counts.

## 3. Model

Compact ConvNeXt defined in `modules.py`: depths (2, 2, 4, 2), widths (32, 64, 128, 256), LayerScale (init 1e-6), stochastic depth 0.1, **1,900,898 parameters**. No pretrained weights.

Input modes (`--input-mode`):
- `single`: one slice, grey value copied to 3 channels.
- `context3`: previous, current and next slice of the *same scan* as the 3 channels (edge slices repeat). Every sample is still one slice with its own label, so evaluation stays per slice.

Training recipe (`train.py`): AdamW, lr 3e-4, weight decay 0.03, batch 8, 30 epochs, 1 warm-up epoch then per-step cosine decay, EMA of weights (0.999, the EMA weights are validated and saved), label smoothing 0.1, mild affine + noise augmentation (no left-right flips), optional `--intensity-augment`. **The checkpoint is chosen by validation slice macro-F1 at threshold 0.5.** The final model is an ensemble of three seeds (42, 43, 44) that averages the AD probability.

## 4. Baseline comparison (identical official split, same test slices)

Baseline: `SimpleCNN`, three conv layers (16/32/64 channels), **23,714 parameters**, single-slice input, same split, same training recipe, one seed. The best epoch was picked on validation (epoch 4, validation macro-F1 0.680).

| | ConvNeXt (3-seed ensemble, context3) | SimpleCNN |
|---|---|---|
| Parameters | 1,900,898 | 23,714 |
| Validation accuracy / AUROC | 76.88 % / 0.858 | 68.91 % / 0.761 |
| **Test accuracy / AUROC** | **75.31 % / 0.847** | 66.46 % / 0.732 |
| AD recall / NC recall (test) | 58.9 % / 91.5 % | 51.1 % / 81.5 % |
| Test accuracy, seen / unseen subjects | 90.4 % / 50.8 % | 75.6 % / 51.6 % |
| ECE (test) / Brier (test) | 0.118 / 0.186 | 0.028 / 0.218 |

Caveats about this comparison:
- The ConvNeXt row bundles three things: architecture, the 3-slice context input and the 3-seed ensemble. The 8.9-point gap is the gap between the two *pipelines*, not an architecture-only effect. I did not run a single-seed, single-slice ConvNeXt on the test folder, to avoid additional test looks.
- Under the ConvNeXt-oriented recipe the small CNN became unstable: from epoch 5 it drifted to predicting NC almost everywhere (AD recall 2.8 % at epoch 30, AUROC still about 0.77). Validation-based checkpoint selection is what rescued it. A CNN-specific recipe might do better; it was not tuned.
- A simple MLP baseline was not run.
- **Unseen subjects are at chance for both models** (50.8 % and 51.6 %), even though the architectures differ a lot. This points at the data (subject/visit identity and acquisition shift) rather than at model capacity.

## 5. Calibration and reject option (confidence-based recommendation)

Rule: refer a slice to a human reader when the model confidence is below a cut-off. The cut-off is chosen **on validation** so that accepted validation accuracy is 90 %. It is then frozen and applied to test.

| ConvNeXt ensemble | Cut-off | Coverage | Accuracy of accepted slices | AD slices missed and not referred |
|---|---|---|---|---|
| Validation | 0.923 | 52.7 % | 90.0 % | 196 |
| Test | 0.923 | 56.8 % | **84.6 %** | 757 of 4,460 (17 %) |

| SimpleCNN | Cut-off | Coverage | Accuracy of accepted slices |
|---|---|---|---|
| Validation | 0.822 | 10.7 % | 90.1 % |
| Test | 0.822 | 12.2 % | 79.1 % |

![Reliability diagram](figures/convnext_reliability.png)
![Accuracy vs coverage](figures/convnext_coverage_accuracy.png)

Recommendation:
1. **Use confidence only as a triage aid, never as a safety guarantee.** The model is over-confident throughout (reliability curve below the diagonal). 959 of the ConvNeXt's test errors have confidence above 0.9.
2. The cut-off that gave 90 % on validation gave 84.6 % on test. A reject threshold must be **re-calibrated on data from the target population** before use, because validation does not reflect the distribution shift.
3. Even with the rule, 17 % of AD slices are neither detected nor referred. This model should not be used to rule disease out.
4. The SimpleCNN is better calibrated (ECE 0.028) only because it is rarely confident, so its reject rule keeps just 12 % of slices and is of little use.

## 6. Where the model fails (subgroups of the official test, ConvNeXt ensemble)

![Test accuracy by group](figures/convnext_test_accuracy_by_group.png)

These breakdowns were made **after** the final test run; they are descriptive, and they were not used to change anything.

| Group | Slices | Accuracy |
|---|---|---|
| AD / NC | 4,460 / 4,540 | 58.9 % / 91.5 % |
| Sequence family: IR-FSPGR / MPRAGE / other* | 2,000 / 6,580 / 420 | 72.4 % / 75.5 % / 86.9 % |
| Scan year ≤2014 / 2015–2018 / ≥2019 | 1,980 / 720 / 6,260 | 90.4 % / **39.6 %** / 74.8 % |
| Subject seen / unseen in training folder | 5,560 / 3,440 | 90.4 % / 50.8 % |

\* "other" is 340 slices of IR-SPGR variants (the analysis script groups sequence names by the substrings "FSPGR" and "MPRAGE", so IR-SPGR without the F falls here) plus 80 slices with no sequence name in the metadata. Scan year comes from the date in the ADNI raw filename; "unknown" year (40 slices) is not shown in the table.

Observations (correlational, not proven causes):
- The model is biased towards NC.
- Sequence family does **not** explain the failure: IR-FSPGR and MPRAGE differ by only 3 points.
- 2015–2018 scans are the worst for both models (39.6 % and 33.8 %), but only 720 slices (about 36 scans) fall there, so this estimate is noisy.
- The SimpleCNN's decision changes with the sequence. On IR-FSPGR it labels AD well (recall 81.7 %) but NC badly (52.1 %), while on MPRAGE it is the reverse (AD 37.5 %, NC 89.8 %). That looks like a shortcut on acquisition appearance. The ConvNeXt is stable across sequences (AD recall 52.9 % vs 57.2 %).

## 7. Failure autopsies

Method: for each error type the most *confidently wrong* **scan** is selected (scan probability = mean over its slices), which is deliberately the worst case rather than a typical error. The analysis script also writes a montage of 5 evenly spaced slices per scan; it is kept locally because it shows MRI images. Raw scan IDs are kept out of the repository (`failure_cases_LOCAL_ONLY.csv`, git-ignored).

| # | Model | Truth → predicted | Scan P(AD) | Sequence, year | Subject seen in training? |
|---|---|---|---|---|---|
| 1 | ConvNeXt | AD → NC | 0.050 | Accelerated Sagittal MPRAGE, 2018 | no |
| 2 | ConvNeXt | AD → NC | 0.050 | Accelerated Sagittal MPRAGE, 2019 | no |
| 3 | ConvNeXt | AD → NC | 0.052 | Accelerated Sagittal MPRAGE, 2020 | no |
| 4 | ConvNeXt | NC → AD | 0.879 | Accelerated Sagittal IR-FSPGR, 2021 | yes |
| 5 | ConvNeXt | NC → AD | 0.809 | Accelerated Sagittal MPRAGE, 2019 | yes |
| 6 | SimpleCNN | AD → NC | 0.0004 | Accelerated Sagittal MPRAGE, 2018 | no |

Cases 1–3 (missed AD, high confidence): all are unseen subjects scanned in 2018–2020 with the same MPRAGE protocol, which is the dominant protocol in the test set (3,960 of 9,000 slices). *Hypothesis:* the model has learnt appearance cues of subjects it saw rather than disease features, so a new subject with mild changes looks "normal". *How to check:* leave-subject-out validation by scan year; compare with a model trained with strict separation. *Mitigation:* strict patient-level training data, per-scan intensity normalisation, a reject rule that is re-calibrated per site.

Cases 4–5 (false alarm on NC): both are subjects that also appear in the training folder. *Hypothesis:* the same person was labelled differently at another visit, or the subject's visit looks like a training AD scan (disease progression or diagnosis conversion is possible in ADNI, label noise cannot be excluded). *How to check:* look up the label of the other visit in the metadata. *Mitigation:* use a per-visit label audit and do not train on conflicting subjects.

Case 6 (SimpleCNN): P(AD)=0.0004 on a true AD scan, a confidence of 99.96 % on the wrong class. This is the strongest argument against using confidence from such a small model for rejection.

Clinical interpretation of the images is deliberately **not** attempted; there is no radiologist input in this project.

## 8. Feasibility review and why 0.80 was not reached

- Spec target: at least 0.80 test accuracy on AD vs NC. Under a normal-score (equal-variance binormal) approximation, 80 % accuracy needs an AUROC of about 0.88. The best result is 0.847 on the leaky official split and about 0.73 on the strict split.
- To reach 0.80 on the official test while seen subjects stay at 90.4 %, slices of the 115 unseen subjects would have to go from 50.8 % to about 63 %. With the improvements tried (augmentation, 2.5-D input, EMA, cosine schedule, ensembling), unseen-subject accuracy on the official split stayed near chance.
- One idea that is still legitimate (not tried): a domain-aware validation set built from later-year and IR-FSPGR training subjects with per-scan intensity normalisation, then a single re-test with full disclosure. My estimate of success is below 30 %.
- Midpoint feasibility check-off: **[TODO: fill in date/tutor name, or state that it was not completed]**.
- Ed discussion with staff about subject overlap: **[TODO: thread number / link]**.

## 9. Resource profile (NVIDIA A100, Rangpur `comp3710` partition)

| | ConvNeXt | SimpleCNN |
|---|---|---|
| Parameters (one model) | 1,900,898 | 23,714 |
| GPU forward time per slice, batch 16 | 0.587 ms for the **3-model ensemble** | 0.080 ms |
| Peak GPU memory, inference | 100 MB with **3 models** loaded | 117 MB |
| Peak GPU memory, training one model (batch 8, includes EMA copy and validation) | 225 MB | 106 MB |

Notes: inference numbers were measured on the 9,000 test slices and cover the GPU forward pass only (JPEG decoding and data loading excluded). A single ConvNeXt was not timed separately; it should cost roughly a third of the ensemble figure. Peak memory is dominated by activation size, not parameter count: the CNN's first convolution runs at the full 224 × 224 resolution, whereas the ConvNeXt stem down-samples by 4 straight away. **[TODO: add wall-clock training time per run from the Slurm logs.]**

## 10. Reproducing the results

```bash
# dependencies (exact versions used)
# python 3.10.21, torch 2.0.1+cu117, torchvision 0.15.2+cu117,
# numpy 1.26.4, matplotlib 3.10.9, Pillow 12.3.0

# train one ConvNeXt (official split, context input)
python -u train.py --model convnext --split-mode official --input-mode context3 \
  --epochs 30 --batch-size 8 --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 \
  --augment --scheduler cosine --warmup-epochs 1 --ema-decay 0.999 --label-smoothing 0.1 \
  --num-workers 4 --seed 42 --run-name official_cosine_ema_context3_s42

# baseline
python -u train.py --model cnn --split-mode official --epochs 30 --batch-size 8 \
  --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 --augment \
  --scheduler cosine --warmup-epochs 1 --ema-decay 0.999 --label-smoothing 0.1 \
  --num-workers 4 --run-name cnn_official_cosine_ema_s42

# analysis of the frozen models (runs the official test once)
python -u analyze.py --confirm-final-test --num-workers 4 --output-dir eval_analysis_convnext \
  --checkpoint checkpoints/convnext/official_cosine_ema_context3_s4{2,3,4}/best.pth
python -u analyze.py --model cnn --confirm-final-test --num-workers 4 --output-dir eval_analysis_cnn \
  --checkpoint checkpoints/cnn/cnn_official_cosine_ema_s42/best.pth
```

Files: `modules.py` (models), `dataset.py` (splits, input modes, augmentation), `train.py`, `predict.py` (inference, ensemble, official-test breakdown by seen/unseen subjects), `analyze.py` (calibration, reject rule, subgroup analysis, failure cases, resource profile), `mri_intensity.py` and the `test_*.py` unit tests, Slurm scripts `run_official_split.slurm` and `run_recipe.slurm`. **[TODO: check this list against the folder and describe `baseline.py` and `diagnose_development.py` if they are kept.]**

Datasets, model checkpoints, logs and raw scan IDs are not committed. The `eval_*` output folders are git-ignored; the summary figures used in this README are copied into `figures/`. The failure-case montage contains MRI images and is not committed.

## 11. AI-tool usage disclosure

I used ChatGPT, Codex and Claude (Claude Code) during this project for code review, writing and refactoring code, designing the experiments, debugging Slurm jobs and drafting this README. I ran all training and evaluation myself on Rangpur and checked the reported numbers against the job outputs. **[TODO: edit to match exactly what you did and what you wrote yourself.]**
