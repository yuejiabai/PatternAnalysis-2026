# ConvNeXt (from scratch) for AD vs CN classification on ADNI brain MRI

COMP3710 Pattern Recognition · Hard difficulty · Student s4985919 · Folder `recognition/ConvNeXt_Bai_s4985919/`

> Research prototype, not a clinical tool. **The 0.80 test-accuracy target was not reached** (best slice-level test accuracy 75.31 %, on a split that shares subjects with the test folder; 65.78 % with strict subject separation).

## 1. Problem and algorithm

**Problem.** Alzheimer's disease (AD) causes progressive cortical and hippocampal atrophy and ventricular enlargement that is visible on T1-weighted structural MRI. The task is to classify 2D MRI slices from the course ADNI data as AD or cognitively normal (CN; the course folder calls this class `NC`). The teaching team confirmed that the target metric is **slice-level accuracy on the official test folder** and that pretrained weights are not allowed. The open engineering question for this project is whether the model's confidence can be trusted to send ambiguous cases to a human reader, and whether a modern architecture justifies its cost over a simple baseline.

**How it works.** ConvNeXt [1] modernises a ResNet-style convolutional network with ideas from vision transformers. A 4 × 4, stride-4 "patchify" stem turns the 224 × 224 input into a 56 × 56 feature map. Four stages follow, separated by 2 × 2 stride-2 down-sampling layers. Each ConvNeXt block applies a 7 × 7 depthwise convolution (large spatial context, few parameters), LayerNorm, an inverted bottleneck (1 × 1 expansion to 4× width, GELU, projection back), a learnable per-channel LayerScale and stochastic depth on the residual branch. Global average pooling, LayerNorm and a linear layer give two logits (AD, CN). This implementation (`modules.py`) is a compact version written from scratch for this dataset size: depths (2, 2, 4, 2), widths (32, 64, 128, 256), LayerScale initialised to 1e-6, stochastic depth up to 0.1, **1,900,898 parameters** and no pretrained weights. The baseline (`baseline.py`) is a 3-layer CNN with 23,714 parameters.

![Validation curves of one ConvNeXt run](figures/convnext_training_curves.png)


## 2. Headline results

| Setting | Model | Test slice accuracy | AUROC (AD) | What it measures |
|---|---|---|---|---|
| **Official split** (course folders as given) | ConvNeXt, 3-seed ensemble, 3-slice context input | **75.31 %** | 0.847 | Not subject-independent: 216 of the 331 test subjects have another visit in the training folder |
| **Strict split** (those 216 subjects removed from training) | ConvNeXt, single model, 8 epochs, earlier recipe, AD threshold 0.67 chosen on validation | **65.78 %** | 0.73 | Generalisation to new subjects |
| Official split, same test slices | SimpleCNN baseline | 66.46 % | 0.732 | Baseline under identical evaluation |

Why two splits: the course folders are separated by *scan* (ADNI Image ID), not by *subject*. Mapping image IDs to subject IDs with the course metadata shows 216 subjects with different visits in both folders. I reported this to the teaching team on Ed. The tutor (Zhao Wang, 18 Oct 2026, thread now private) advised using **the provided split as the primary result** and reporting the patient-disjoint results as a clearly labelled comparison, which is what this report does. The official split is therefore the primary result, but it is not subject-independent; the strict split is the better estimate for completely new patients.

Breaking the official-split ensemble down by whether the test subject was seen in training shows where the score comes from:

| Test subjects | Slices | Share AD | Accuracy | AD recall | CN recall |
|---|---|---|---|---|---|
| Seen in training folder (216) | 5,560 | 37 % | **90.4 %** | 88.1 % | 91.9 % |
| Not seen (115) | 3,440 | 69 % | **50.8 %** | **33.3 %** | 90.2 % |

On new subjects the model still recognises CN scans, but it misses two thirds of AD slices. Because the unseen group is mostly AD, its overall accuracy falls to 50.8 %.

**Secondary, aggregated levels** (reported by `predict.py` for every evaluation since the start of the project, not the target metric): averaging the AD probability over the 20 slices of a scan, or over all scans of a subject, gives test accuracy 77.33 % per scan and 82.18 % per subject for the official-split ensemble (validation: 80.19 % and 84.07 %), and 66.44 % / 70.09 % for the strict-split model. The tutor confirmed that the assessed metric is slice-level, so these numbers are context only and do not mean the 0.80 target was met.

**Test-set use.** The official test folder was already evaluated during development, before this write-up (Section 9). For the final models, checkpoints (best epoch), the AD threshold (0.5, fixed before training) and the reject rule were chosen on validation only, and nothing was changed after looking at the test breakdowns in Sections 5–7. The 3-slice input was preferred because it scored higher on validation than single-slice input (76.55 % vs 75.56 % at seed 42; Section 9). The ensemble is used because its validation accuracy (76.88 %) is higher than that of each of its three members (76.55–76.79 %). I cannot rule out that test accuracies of single seeds were also viewed during development; the saved logs in `run_logs/evaluation/` contain test results only for the runs listed in Section 9 and for the ensemble.

## 3. Data, pre-processing and splits

- Data: `/home/groups/comp3710/ADNI/AD_NC` (`train/` and `test/`, classes `AD` and `NC`) and the metadata file `/home/groups/comp3710/ADNI/meta_data_with_label.json`. Each scan contributes 20 JPEG slices named `scanID_sliceNo.jpeg`; the prefix is the ADNI Image ID of one scan, not a patient. The subject ID (e.g. `137_S_4672`) is taken from the metadata `raw` path with the regex `\d{3}_S_\d{4}`.
- Labels: AD = class 0 (positive class for recall and AUROC), CN = class 1, following ImageFolder order. The mapping is asserted in every script.
- Pre-processing (`dataset.py`): resize to 224 × 224, convert to greyscale, replicate to 3 channels, scale to [0, 1] and normalise with mean = std = 0.5, so pixels lie in [−1, 1].
- Input modes: `single` copies one slice into the 3 channels. `context3` (2.5D) puts the previous, current and next slice **of the same scan** into the 3 channels, with edge slices repeated. Every sample is still one slice with its own label, so evaluation stays per slice.
- Training-only augmentation: random affine (±6°, ±3 % shift, 0.97–1.03 scale, background filled black) and Gaussian noise (σ = 0.01). No left-right flips, because hemispheric asymmetry may carry information. Optional `--intensity-augment` adds a random scale and gamma (0.9–1.1). Validation and test images are never augmented.
- Validation is split **by subject** (20 %, stratified, seed 42) in both modes; code assertions guarantee that no subject is in both training and validation.

| Mode | Training | Validation | Test |
|---|---|---|---|
| `strict` | 279 subjects, 10,080 slices | 70 subjects, 2,620 slices | 331 subjects, 450 scans, 9,000 slices |
| `official` | 452 subjects, 17,380 slices | 113 subjects, 4,140 slices | same |

In strict mode, 8,820 training slices (41 %) are removed because their subject appears in the test folder. Official mode prints a leakage warning (186 training and 30 validation subjects overlap with test). The test folder has 4,460 AD and 4,540 CN slices. `python dataset.py strict` or `python dataset.py official` prints these counts.

## 4. Training

`train.py` imports the model from `modules.py` and the data from `dataset.py`. Final recipe (`run_recipe.slurm`): AdamW (lr 3e-4, weight decay 0.03), batch size 8, 30 epochs, 1 warm-up epoch followed by per-step cosine decay, exponential moving average of the weights (decay 0.999; the EMA weights are validated and saved), label smoothing 0.1 and the augmentation above. After every epoch the model is scored on validation, and `best.pth` is kept when the **validation slice macro-F1 at AD threshold 0.5** improves (ties: AD recall, then validation loss). Every run saves `best.pth`, `best_meta.json` (selection settings, split fingerprints), `last.pth` (resumable state, also on Slurm time-out), `history.json`, `config.json` and `training_curves.png`. The final ConvNeXt is an ensemble of three seeds (42, 43, 44) that averages the AD probability.

## 5. Baseline comparison (official split, identical test slices)

Baseline: `SimpleCNN` (3 conv layers 16/32/64 + ReLU + max-pool, global average pool, linear), single-slice input, same split and same recipe, one seed. Its best epoch was chosen on validation (epoch 4, validation macro-F1 0.680).

| Test metric (slice level, threshold 0.5) | ConvNeXt ensemble | SimpleCNN |
|---|---|---|
| Accuracy | **75.31 %** | 66.46 % |
| AUROC (AD positive) | **0.847** | 0.732 |
| AD precision / recall / F1 | 0.872 / 0.589 / 0.703 | 0.731 / 0.511 / 0.602 |
| CN precision / recall / F1 | 0.694 / 0.915 / 0.789 | 0.629 / 0.815 / 0.710 |
| Macro-F1 | **0.746** | 0.656 |
| Confusion (TP, FN, FP, TN; AD positive) | 2625, 1835, 387, 4153 | 2279, 2181, 838, 3702 |
| Validation accuracy / AUROC / macro-F1 | 76.88 % / 0.858 / 0.762 | 68.91 % / 0.761 / 0.680 |
| Test accuracy, seen / unseen subjects | 90.4 % / 50.8 % | 75.6 % / 51.6 % |

![Test confusion matrix, ConvNeXt](figures/convnext_test_confusion.png)
![ROC curves, ConvNeXt](figures/convnext_roc.png)

Caveats:
- The ConvNeXt column bundles architecture, 3-slice input and a 3-model ensemble, so the 8.9-point gap is a *pipeline* difference, not an architecture-only effect. A single-seed, single-slice ConvNeXt was not evaluated on test, to avoid extra test looks.
- The recipe was tuned for ConvNeXt. Under it the CNN was unstable: from epoch 5 it drifted towards predicting CN almost everywhere (AD recall 2.8 % at epoch 30, validation AUROC still about 0.77). Validation-based checkpoint selection kept the epoch-4 weights. A CNN-specific recipe was not tuned.

![CNN validation curves](figures/cnn_training_curves.png)
- Both models are about equally poor on unseen subjects (50.8 % vs 51.6 %), so the architecture does not fix the subject-generalisation problem.

## 6. Open dilemma: confidence, calibration and a reject option

**Does confidence track correctness?** Partly. ConvNeXt's mean confidence is 0.888 on correct and 0.811 on wrong test slices, and accepted-slice accuracy rises as fewer slices are accepted (coverage curve). But it is over-confident everywhere (reliability curve below the diagonal, test ECE 0.118), and **959 of its 2,222 test errors have confidence ≥ 0.9**. The CNN has a lower average calibration error (ECE 0.028; Brier 0.218 vs 0.186) mainly because it is rarely confident (mean confidence 0.691 correct vs 0.651 wrong), so its confidence barely separates right from wrong. Worse, its *few* very confident test predictions are mostly wrong: in its reliability diagram, test slices with confidence above about 0.95 are correct far less than half the time, and its accepted-slice accuracy on test collapses at low coverage. A low average ECE therefore hides the most dangerous errors.

![Reliability diagram, SimpleCNN](figures/cnn_reliability.png)

![Reliability diagram, ConvNeXt](figures/convnext_reliability.png)
![Accuracy vs coverage, ConvNeXt](figures/convnext_coverage_accuracy.png)

**Reject rule.** A slice is referred to a human reader when confidence (probability of the predicted class) is below a cut-off. The cut-off was fixed in advance as the lowest one giving 90 % accepted accuracy **on validation**, then applied unchanged to test.

| | Cut-off | Coverage (decided automatically) | Accuracy of accepted slices | Accuracy of referred slices | AD slices missed and not referred |
|---|---|---|---|---|---|
| ConvNeXt, validation | 0.923 | 52.7 % | 90.0 % | 62.2 % | 196 of 1,860 |
| ConvNeXt, test | 0.923 | 56.8 % | **84.6 %** | 63.1 % | **757 of 4,460 (17 %)** |
| CNN, validation | 0.822 | 10.7 % | 90.1 % | 66.4 % | 19 of 1,860 |
| CNN, test | 0.822 | 12.2 % | 79.1 % | 64.7 % | 222 of 4,460 |

The rule transfers poorly: 90 % on validation became 84.6 % on test, and test coverage went *up*, i.e. the model is more confident on the shifted test data while being less accurate.

## 7. Where and why it fails (official test, ConvNeXt ensemble)

These breakdowns were made after the final test run. They describe the results; nothing was tuned on them.

![Test accuracy by group](figures/convnext_test_accuracy_by_group.png)

| Group | Slices | Share AD | Accuracy | AD recall | CN recall |
|---|---|---|---|---|---|
| Sequence IR-FSPGR | 2,000 | 50 % | 72.4 % | 52.9 % | 91.9 % |
| Sequence MPRAGE | 6,580 | 47 % | 75.5 % | 57.2 % | 91.3 % |
| Sequence other* | 420 | 95 % | 86.9 % | 86.3 % | 100 % |
| Scan year ≤ 2014 | 1,980 | **100 %** | 90.4 % | 90.4 % | – |
| Scan year 2015–2018 | 720 | **100 %** | 39.6 % | 39.6 % | – |
| Scan year ≥ 2019 | 6,260 | 28 % | 74.8 % | 31.7 % | 91.4 % |

\* "other" = 340 slices of IR-SPGR variants (the script groups names by the substrings "FSPGR" and "MPRAGE") plus 80 slices with no sequence name. Scan year is the date in the ADNI raw filename; 40 slices have no year.

Findings (correlational, not proven causes):
1. **Scan year and label are confounded in the test folder.** Every test scan from before 2019 is AD, and every CN scan is from 2019 or later. AD recall falls from 90 % (≤ 2014) to 40 % (2015–2018) and 32 % (≥ 2019). Year, subject overlap (seen subjects: AD recall 88 %; unseen: 33 %) and disease stage were not cross-tabulated here and may overlap, so "acquisition era" is a plausible shortcut but not demonstrated.
2. **Sequence family does not explain the failure for ConvNeXt**: IR-FSPGR and MPRAGE give similar AD and CN recall.
3. **The CNN does show a sequence shortcut.** On IR-FSPGR its AD recall is 81.7 % but CN recall 52.1 %; on MPRAGE it is the reverse (37.5 % and 89.8 %), with similar class balance in both. ConvNeXt is stable across sequences.
4. Both models are biased towards CN: most errors are missed AD (1,835 FN vs 387 FP for ConvNeXt).

### Failure autopsies

For each error type the most *confidently wrong* **scans** were selected (scan probability = mean of its 20 slices). These are deliberately worst cases, not average errors. Raw scan IDs stay local (`failure_cases_LOCAL_ONLY.csv`, git-ignored), and MRI images are not published.

| # | Model | Truth → predicted | Scan P(AD) | Sequence, year | Subject seen in training? |
|---|---|---|---|---|---|
| 1 | ConvNeXt | AD → CN | 0.050 | Accelerated Sagittal MPRAGE, 2018 | no |
| 2 | ConvNeXt | AD → CN | 0.050 | Accelerated Sagittal MPRAGE, 2019 | no |
| 3 | ConvNeXt | CN → AD | 0.879 | Accelerated Sagittal IR-FSPGR, 2021 | yes |
| 4 | ConvNeXt | CN → AD | 0.809 | Accelerated Sagittal MPRAGE, 2019 | yes |
| 5 | SimpleCNN | AD → CN | 0.0004 | Accelerated Sagittal MPRAGE, 2018 | no |

- **Cases 1–2 (missed AD at 95 % confidence).** New subjects, recent MPRAGE scans: exactly the group with 32 % AD recall. *Trigger hypothesis:* the network partly learned subject- or era-specific appearance instead of disease features, and recent AD scans of new subjects look like the recent CN scans it saw. *Check:* train with strict separation and compare AD recall by year on validation. *Mitigation:* strict subject-level training data, a validation set that matches the deployment era, never auto-clearing CN.
- **Cases 3–4 (false alarm on CN).** Both subjects have another visit in the training folder. *Trigger hypothesis:* the subject was AD-labelled at another visit, or looks like one (diagnoses can change between ADNI visits). *Check:* compare with the labels of the subject's other visits in the metadata. *Mitigation:* audit per-visit labels before training.
- **Case 5 (CNN, 99.96 % confident and wrong).** Shows that a small model's confidence can also be unreliable, even though its average calibration looks good.

No clinical reading of the images is attempted; there was no radiologist input.

## 8. Resource profile

Measured on an NVIDIA A100 (Rangpur `comp3710` partition) with `torch.cuda.max_memory_allocated` and CUDA-synchronised timing.

| | ConvNeXt | SimpleCNN |
|---|---|---|
| Parameters (one model) | 1,900,898 | 23,714 |
| GPU forward time per slice, batch 16 | 0.587 ms for the **3-model ensemble** | 0.080 ms |
| Peak GPU memory, inference | 100 MB with **3 models** loaded | 117 MB |
| Peak GPU memory, training one model (batch 8, incl. EMA copy and validation) | 225 MB | 106 MB |
| Wall-clock training time, 30 epochs incl. validation every epoch (Slurm elapsed time) | 53–56 min per seed (4 runs: 52:37, 55:29, 53:26, 53:26); about 2.7 h for the 3-seed ensemble | 21 min |

Inference timing covers the GPU forward pass only (no JPEG decoding or loading) over the 9,000 test slices. A single ConvNeXt was not timed separately; it should cost roughly a third of the ensemble figure. Peak memory follows activation size more than parameter count: the CNN's first convolution runs at full 224 × 224 resolution, while the ConvNeXt stem down-samples by 4 immediately. Both models fit easily on a 16 GB P100.

## 9. Experiment history

All runs, read from each run's `config.json`, `history.json` and `best_meta.json`. These files are written automatically by `train.py` and are committed unchanged in [`run_logs/`](run_logs/) (with `run_logs/summary.csv`), so every number below can be checked against the original logs; `collect_runs.py` gathers them and draws the figure.

![Validation accuracy per epoch for all 13 runs](figures/experiment_history.png)

 "Best val acc" is the highest validation slice accuracy at AD threshold 0.5 over all epochs (slightly optimistic, because it is a maximum); "saved" is the validation accuracy of the checkpoint actually kept by the selection rule (validation macro-F1), where the run recorded it. Strict and official runs use **different validation subjects** (70 vs 113), so accuracies are only comparable within the same split.

| # | Run | Split | Change tested | Epochs | Best val acc (epoch) | Saved checkpoint | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `convnext_run01` | strict | first ConvNeXt: Adam, lr 1e-3, no augmentation | 3 | 66.37 % (2) | – | starting point |
| 2 | `convnext_lr3e4_run01` | strict | lower learning rate 3e-4 | 5 | 63.66 % (3) | – | no gain |
| 3 | `convnext_aug_adamw_run01` | strict | AdamW (wd 0.03) + geometric/noise augmentation | 8 | 77.75 % (7) | – | large gain; kept; tested once (below) |
| 4 | `convnext_cosine_run01` | strict | + cosine learning-rate schedule (earlier implementation) | 12 | 77.90 % (9) | – | +0.15 points, within noise |
| 5 | `convnext_patient_balanced_run01` | strict | + patient-balanced sampling | 8 | 76.37 % (6) | – | no gain |
| 6 | `convnext_intensity_adamw_run01` | strict | + intensity (scale/gamma) augmentation | 8 | 77.71 % (7) | 77.71 % (epoch 7) | no gain |
| 7 | `cnn_run01` | strict | SimpleCNN baseline, Adam, lr 1e-3 | 3 | 70.92 % (2) | – | baseline |
| 8 | `convnext_official_aug_adamw_run01` | official | run 3's recipe on the official split | 8 | 73.04 % (8) | 72.15 % (epoch 6) | evaluated on test during development (71.82 % slice accuracy); superseded |
| 9 | `official_cosine_ema_single_s42` | official | final recipe (30 epochs, warm-up + cosine, EMA, label smoothing), single-slice input | 30 | 75.56 % (28) | 75.56 % (epoch 28) | – |
| 10 | `official_cosine_ema_context3_s42` | official | final recipe, 3-slice context input | 30 | 76.59 % (24) | 76.55 % (epoch 21) | context3 better than single on validation; used |
| 11 | `official_cosine_ema_context3_s43` | official | same, seed 43 | 30 | 76.67 % (20) | 76.67 % (epoch 20) | used |
| 12 | `official_cosine_ema_context3_s44` | official | same, seed 44 | 30 | 76.79 % (16) | 76.79 % (epoch 16) | used |
| 13 | `cnn_official_cosine_ema_s42` | official | SimpleCNN with the final recipe | 30 | 68.91 % (4) | 68.91 % (epoch 4) | baseline (Section 5) |

The 3-seed ensemble of runs 10–12 reaches 76.88 % validation accuracy, slightly above each member (76.55–76.79 %).

**Test result of run 3** (the strict-split model; AD threshold 0.67 chosen on validation, where slice/scan/patient accuracy was 0.788 / 0.855 / 0.886): test slice / scan / patient accuracy **0.658 / 0.664 / 0.701**, patient-level AUROC 0.770 and AD recall 0.473 (69 of 131 AD patients called CN). The large validation–test drop (0.79 → 0.66 at slice level) is a further sign of the shift between the training folder and the test folder.

Other development analyses (no test data): `diagnose_development.py` audits subject overlap, mixed labels across visits and split fingerprints, and scores the fixed run-3 model (threshold 0.67) on its own training and validation subjects. Slice accuracy falls step by step as the data move further from what the model saw: **87.1 %** on training subjects (AUROC 0.941), **78.8 %** on new validation subjects (AUROC 0.863) and **65.8 %** on the test folder (AUROC 0.732). The first gap is ordinary over-fitting to training subjects; the second is the train-folder-to-test-folder shift discussed in Section 7. No validation subject had mixed AD/CN labels across visits.

`test_brightness_sensitivity.py` measures how predictions change when validation images are made 10 % darker or brighter (run 3, AD threshold 0.67, a fixed sample of 280 validation slices from all 70 validation subjects and 122 scans; 95 % intervals from a 200-sample bootstrap over subjects):

| Brightness factor | Accuracy | AD recall | CN recall | AUROC | Slices whose prediction flips |
|---|---|---|---|---|---|
| 1.0 (original) | 77.5 % | 76.4 % | 77.9 % | 0.851 | – |
| 0.9 (10 % darker) | 77.9 % (change +0.4, CI −3.9 to +4.7) | 72.2 % | 79.8 % | 0.848 | 8.9 % (CI 5.7–12.9 %) |
| 1.1 (10 % brighter) | 73.6 % (change **−3.9**, CI −7.1 to −0.7) | 75.0 % | 73.1 % (change −4.8, CI −8.6 to −1.3) | 0.840 | 6.1 % (CI 3.2–9.3 %) |

A global brightness change of only 10 % flips 6–9 % of slice decisions, and brightening lowers accuracy significantly, mostly by calling more CN slices AD. The model is therefore sensitive to intensity differences of the size that can occur between scanners and protocols, which supports the acquisition-shift explanation in Section 7 and the recommendation of per-scan intensity normalisation. Training with random intensity (scale/gamma) augmentation (run 6) did not raise validation accuracy (77.71 % vs 77.75 %); its effect on this sensitivity was not measured.

## 10. Feasibility review (midpoint check-off) and outcome

**[TODO: replace this summary by the exact one-page review shown to the tutor, and add the check-off date and tutor name.]**

1. **User need, scope, acceptance criteria.** A triage aid that labels AD vs CN slices and routes uncertain cases to a specialist. Acceptance criteria: (a) strict subject-level separation, (b) slice-level test accuracy ≥ 0.80, (c) comparison with a simple baseline on identical splits, (d) per-class metrics and a confidence/reject analysis, (e) training and inference fit a 16 GB P100 within 2-hour Slurm jobs.
2. **Model choice and course concepts.** Compact from-scratch ConvNeXt [1] vs a simple CNN; depthwise convolutions, LayerNorm, inverted bottlenecks and stochastic depth; slice-level training with optional probability averaging per scan and per patient.
3. **Preliminary evidence.** Subject-level split 279 / 70 / 331 subjects; CNN and ConvNeXt train and predict end to end; first test accuracy below the target.
4. **Risks, budget, fallback.** Few independent subjects, validation–test gap and missed AD cases; 2-hour GPU jobs with resumable checkpoints; fallback: CNN plus mandatory specialist review.

| Criterion | Outcome |
|---|---|
| (a) strict separation | Met in `strict` mode (65.78 %); the 75.31 % official-split result violates it and is labelled as such |
| (b) accuracy ≥ 0.80 | **Not met** |
| (c) baseline comparison | Met (Section 5) |
| (d) per-class metrics, reject analysis | Met (Sections 5–6) |
| (e) resources | Met (Section 8) |

**Why 0.80 was not reached.** Under an equal-variance normal approximation, 80 % accuracy needs an AUROC of about 0.88; the best AUROC is 0.847 on the leaky split and about 0.73 on the strict split. To reach 80 % on the official test while seen subjects stay at 90.4 %, accuracy on the 115 unseen subjects would have to rise from 50.8 % to about 63 %. Augmentation, 2.5D input, EMA, a cosine schedule and ensembling did not move unseen-subject performance. One untried, legitimate idea is a validation set matched to the test era (recent IR-FSPGR/MPRAGE subjects from the training folder) with per-scan intensity normalisation, followed by a single, disclosed re-test.

## 11. Recommendation to the project manager

1. **Do not deploy, and do not use the model to rule out AD.** On new subjects it misses about two thirds of AD slices. Even with the reject rule, 17 % of AD test slices are neither detected nor referred.
2. **Use confidence only for triage, and re-calibrate on local data.** The 90 % validation operating point gave 84.6 % on test; the threshold must be set on data from the target site and era.
3. **ConvNeXt vs baseline.** ConvNeXt is clearly better on the same test (+8.9 points accuracy, +0.115 AUROC, half the CN false alarms) at a cost that is negligible in absolute terms (≈ 0.2 ms per slice per model, < 250 MB GPU memory). The added complexity is justified, but it does not solve the main problem, which is generalisation to new subjects and newer scans.
4. **Next steps before any further decision:** a test set with strict subject separation and balanced scan years; scan- or subject-level aggregation; per-scan intensity normalisation; prospective calibration.

## 12. Usage and reproducibility

**Environment** (Rangpur conda env `torch117`): Python 3.10.21, torch 2.0.1+cu117, torchvision 0.15.2+cu117, numpy 1.26.4, matplotlib 3.10.9, Pillow 12.3.0. Seeds are fixed (Python, torch, data order); GPU convolutions are not forced to be deterministic, so repeated runs can differ slightly.

```bash
# split summary (no training)
python dataset.py official

# train one ConvNeXt (repeat with --seed 43 / 44); or: sbatch --export=ALL,INPUT_MODE=context3,SEED=42 run_recipe.slurm
python -u train.py --model convnext --split-mode official --input-mode context3 \
  --epochs 30 --batch-size 8 --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 \
  --augment --scheduler cosine --warmup-epochs 1 --ema-decay 0.999 --label-smoothing 0.1 \
  --num-workers 4 --seed 42 --run-name official_cosine_ema_context3_s42

# baseline
python -u train.py --model cnn --split-mode official --epochs 30 --batch-size 8 \
  --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 --augment \
  --scheduler cosine --warmup-epochs 1 --ema-decay 0.999 --label-smoothing 0.1 \
  --num-workers 4 --run-name cnn_official_cosine_ema_s42

# inference with the saved ensemble (validation; add --split test --confirm-final-test for the frozen test run)
python -u predict.py --split validation --num-workers 4 \
  --checkpoint checkpoints/convnext/official_cosine_ema_context3_s4{2,3,4}/best.pth

# calibration, reject rule, subgroup and failure analysis, resource profile, figures
python -u analyze.py --confirm-final-test --num-workers 4 --output-dir eval_analysis_convnext \
  --checkpoint checkpoints/convnext/official_cosine_ema_context3_s4{2,3,4}/best.pth
python -u analyze.py --model cnn --confirm-final-test --num-workers 4 --output-dir eval_analysis_cnn \
  --checkpoint checkpoints/cnn/cnn_official_cosine_ema_s42/best.pth

# unit tests (synthetic data only, CPU)
python -m unittest -v test_mri_intensity test_diagnose_development test_brightness_sensitivity_unit
```

Example output of `analyze.py` for the ConvNeXt ensemble:

```
Model convnext x3; split official; input context3; class threshold 0.5 (from best_meta.json)
VALIDATION acc=0.7688 AUROC=0.8580 ECE=0.0885 Brier=0.1643 conf(correct)=0.882 conf(wrong)=0.775 wrong@conf>=0.9=299
TEST       acc=0.7531 AUROC=0.8467 ECE=0.1176 Brier=0.1857 conf(correct)=0.888 conf(wrong)=0.811 wrong@conf>=0.9=959
Reject rule (chosen on validation for 90% accepted accuracy): confidence >= 0.92287
  validation coverage=0.527 accepted_acc=0.9001 referred=1957 AD missed & not referred=196
  test       coverage=0.568 accepted_acc=0.8457 referred=3886 AD missed & not referred=757
```

### Files

| File | Purpose |
|---|---|
| `modules.py` | ConvNeXt (LayerNorm2d, DropPath, ConvNeXtBlock, ConvNeXt); PyTorch only |
| `baseline.py` | SimpleCNN baseline |
| `dataset.py` | Subject IDs from metadata, strict/official splits, subject-level validation split, `single`/`context3` input, training-only augmentation |
| `train.py` | Training, validation each epoch, checkpoint selection, resumable checkpoints, training curves; `--test-only` calls `predict.py` |
| `predict.py` | Loads saved weights (one model or an ensemble), slice/scan/patient metrics, guarded test access, seen/unseen subject breakdown |
| `analyze.py` | Calibration (ECE, Brier, reliability), reject rule, ROC, confusion, subgroup accuracy, failure cases, resource profile, figures |
| `mri_intensity.py` | Optional training-only intensity augmentation |
| `diagnose_development.py` | Read-only audit of the development data (subject overlap, mixed labels, split fingerprints) |
| `test_brightness_sensitivity.py` | Validation-only brightness sensitivity experiment (a script, not a unit test) |
| `test_*.py` (others) | Unit tests on synthetic data |
| `run_recipe.slurm`, `run_official_split.slurm` | Slurm jobs for the final recipe and the official split |
| `collect_runs.py` | Copies every run's config/history/selection logs into `run_logs/` and draws `figures/experiment_history.png` |
| `run_logs/` | Original training logs of all 13 runs (settings and per-epoch validation metrics only) |
| `run_logs/evaluation/` | Original `analyze.py` outputs (`analysis_convnext_ensemble.json`, `analysis_cnn.json`): every validation/test number, reject-rule, subgroup, failure-case and resource figure in Sections 2 and 5–8; plus the `predict.py` metric files of earlier evaluations (named `<project>__<output folder>__<file>.json`, e.g. the strict-split test run of Section 9). No scan or subject IDs |
| `figures/` | Plots used in this README (no MRI images) |

Datasets, checkpoints, logs, raw scan IDs and MRI montages are not committed (`.gitignore`).

## 13. Artificial Intelligence Usage Disclosure

**[TODO: check every sentence against what you actually did.]**

- **Tools:** ChatGPT, OpenAI Codex and Anthropic Claude (Claude Code).
- **Tasks they assisted with:** refactoring the data split, training and evaluation code; the split-mode, context-input, EMA/cosine and ensemble options; the analysis script (calibration, reject rule, subgroup analysis); unit tests; debugging Rangpur and Slurm jobs; reviewing results for leakage; and drafting and editing this README.
- **How I verified the output:** I ran all training and evaluation myself on Rangpur and checked every number in this README against job logs and the saved `analysis.json`; subject-level separation is asserted in code; checkpoints carry split and preprocessing fingerprints that `predict.py` and `analyze.py` check before use; unit tests on synthetic data pass on Rangpur; AI suggestions that were wrong (for example an over-simplified reading of the unseen-subject result and an ensemble latency mislabelled as single-model) were found by checking against the data and corrected.

## References

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. CVPR*, 2022, pp. 11976–11986.

[2] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, "Deep networks with stochastic depth," in *Proc. ECCV*, 2016.

[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017 (ECE and reliability diagrams).

[4] Alzheimer's Disease Neuroimaging Initiative (ADNI), http://adni.loni.usc.edu (data used through the course copy on Rangpur).
