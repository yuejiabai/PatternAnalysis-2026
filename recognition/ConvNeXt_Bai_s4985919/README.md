# ADNI Alzheimer's Disease Classification with Compact ConvNeXt

**COMP3710 — Pattern Recognition · Hard Difficulty · Consistent-evaluation review candidate (NOT final submission)**

> This is an engineering research prototype for **AD vs CN classification**, not a clinical diagnostic tool. The latest observed official test accuracy is **below the course target of 0.80**. Do not report validation accuracy as test accuracy.

## 1. Overview and working principle

We classify structural brain MRI slices from the course ADNI data using a compact, **randomly initialised** ConvNeXt and compare it with a simple CNN. ConvNeXt uses a four-stage convolutional hierarchy, 7×7 depthwise convolutions, LayerNorm, GELU, residual connections, and stochastic depth. Its 2D predictions can be combined into scan-level and patient-level scores by averaging AD probabilities. Each slice is a grayscale MRI replicated across three channels to match the model input; these are **not three neighbouring anatomical slices**. The prototype investigates whether advanced architecture and aggregation meaningfully improve AD screening over the baseline, and whether model confidence can safely support human triage.

> Add your own model architecture diagram or training-curve figure here before final submission.

## 2. Data and patient independence

Course dataset: `/home/groups/comp3710/ADNI/AD_NC`; metadata: `/home/groups/comp3710/ADNI/meta_data_with_label.json`.

`dataset.py` maps each scan to its ADNI patient ID using metadata, removes train images associated with official test patients, then assigns **whole patients** to training or validation (seed 42). Neither patients nor their scans/slices are split across partitions. Class mapping is `AD=0` and `NC=1`; the supplied folder label **NC** denotes the cognitively normal class called **CN** in the assignment. Do not change this mapping between scripts.

| Split | Patients | Scans | 2D slices |
|---|---:|---:|---:|
| Train | 279 | 504 | 10,080 |
| Validation | 70 | 131 | 2,620 |
| Official test | 331 | 450 | 9,000 |

216 overlapping patients caused exclusion of 8,820 course-train slices. All 20 slices from a given scan remain together. **Test patient identities** are used solely to prevent overlap, not to train or tune. The dataset is not redistributed.

Input preprocessing: resize to 224×224, convert to grayscale and replicate into 3 channels, convert to tensor, and normalize with mean/std = 0.5. For the augmented ConvNeXt experiment, *training only* additionally applies slight affine transforms and Gaussian noise; validation/test images do not receive augmentations.

### Split modes (strict vs official)

`--split-mode strict` (default) removes every ADNI subject that also has a scan in the official test folder, so the test estimate is subject-independent. `--split-mode official` keeps the course train/test folders exactly as provided. Course staff describe these folders as leak-free; however, the filename prefix is the ADNI **Image ID** of one scan, not the subject, and mapping prefixes to subject IDs through the metadata shows 216 subjects with different visits in both folders. In both modes, validation is split by **subject**. Official-mode test accuracy is reported separately and is **not** a subject-independent estimate.

### Training recipe and input options (official split, validation-selected)

`run_recipe.slurm` trains for 30 epochs with a 1-epoch linear warmup and per-step cosine decay, an exponential moving average (EMA, decay 0.999) of the weights (the EMA weights are validated and saved), and label smoothing 0.1. `--input-mode context3` feeds the previous, current and next slice of the same scan as the three input channels (2.5D); edge slices repeat themselves. Each sample is still one slice with one label, so accuracy remains per slice. Variants are compared on **validation only**; the official test is run once for the variant chosen in advance. `predict.py` accepts several `--checkpoint` files and averages their AD probabilities (seed ensemble); for an official-split test it also reports accuracy separately for test subjects with and without other scans in the course train folder.

## 3. Setup and reproducibility

Developed on Rangpur using Python with PyTorch `2.0.1+cu117` and torchvision `0.15.2+cu117`, P100 GPU (16 GB), with a separate `torch117` conda environment. Install compatible `matplotlib` for plots. **Do not install a newer CUDA-dependent PyTorch build over the working environment without checking P100 compatibility.** Other exact installed package versions must be recorded from the environment before the final submission.

Relevant files:

- `modules.py`: compact ConvNeXt implementation.
- `dataset.py`: MRI loading, patient-level train/validation/test separation, train-only augmentation wrapper.
- `train.py`: CNN or ConvNeXt training, checkpoint/restart, validation metrics and training curves; can delegate explicit test-only evaluation to `predict.py`.
- `predict.py`: inference and slice-, scan-, and patient-level metrics (with explicit official-test opt-in).
- `baseline.py`: simple CNN baseline.

Example on Rangpur, from the project directory **on an allocated GPU node**:

```bash
source ~/miniconda3/bin/activate
conda activate torch117
python -u train.py --model convnext --epochs 8 --batch-size 8 \
  --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 \
  --augment --run-name convnext_aug_adamw_REPRO
```

In new experiments, checkpoint selection is explicitly controlled by `--selection-level` (`slice`, `scan`, or `patient`) and `--selection-threshold` (AD probability cutoff), applied **on validation only**. Defaults are **slice-level macro-F1 at 0.50**, with ties broken by AD recall, then validation loss. Predeclare one level and threshold for a particular run and document the choice; the course has not yet confirmed which level satisfies its 0.80 test target. Use full validation (no `--max-val-batches`) for scan- or patient-level selection. Do **not** change these options when resuming a run.

New runs save `best.pth` (raw state_dict, compatible with old model architecture), `best_meta.json` (chosen level, cutoff, class mapping, preprocessing and split fingerprints), `last.pth` (resumable full training state), `history.json`, `config.json`, and `training_curves.png` in `checkpoints/<model>/<run-name>/`. Training can be resumed with the same model/config flags plus `--resume checkpoints/.../last.pth`, **without** passing `--run-name`. Never overwrite a previous run directory.

Existing, historically evaluated checkpoint on the server (not included in repository):

```
checkpoints/convnext/convnext_aug_adamw_run01/best.pth
```

To reproduce **validation** metrics from the prior checkpoint (legacy model with no `best_meta.json`; the previous 0.67 cutoff was chosen during development):

```bash
python -u predict.py --split validation \
  --checkpoint checkpoints/convnext/convnext_aug_adamw_run01/best.pth \
  --model convnext --threshold 0.67 --output-dir evaluation_results
```

Legacy raw `best.pth` files remain loadable, but the program **warns** that their training-time fingerprints cannot be proven retroactively. For new checkpoints, `predict.py` verifies the model type, class mapping, preprocessing representation, all three split fingerprints, and hashed scan-to-patient mapping against `best_meta.json`, and uses its stored cutoff if `--threshold` is omitted. The default for a legacy checkpoint without metadata is **0.50** (provide `--threshold 0.67` explicitly to reproduce the earlier historical results).

To compare the CNN baseline on the **same validation split**, using its own saved weights:

```bash
python -u predict.py --model cnn --split validation \
  --checkpoint checkpoints/cnn/cnn_run01/best.pth --threshold 0.50 \
  --output-dir evaluation_cnn_validation
```

This is **not a new accuracy claim**: the baseline checkpoint and its on-server path must be verified on Rangpur, and full comparisons still need to be run. Do not run an official test just to complete a README table. For new experiments, run `predict.py` on validation only and compare the specified metric/level.

**The official test has already been evaluated in development and must not be treated as untouched in future model-selection experiments.** Do not choose checkpoints, features, hyperparameters or a threshold by repeatedly querying test. The code supports a guarded test-only path in `predict.py` (and `train.py --test-only`); any later test re-evaluation is exploratory, not another independent held-out estimate. A `--threshold` different from the saved model-selection cutoff raises an explicit warning; test-time threshold adjustment is **not** a valid way to claim improved independent test accuracy.

## 4. Results observed so far

All values below are **observations from previous runs**, not claims that these numbers have been freshly reproduced by this packaging/refactoring. The best previous augmented ConvNeXt used an AD probability threshold of 0.67 selected on validation.

| Accuracy level | Validation | Official test |
|---|---:|---:|
| Slice | 0.7882 | 0.6578 |
| Scan (mean across 20 slices) | 0.8550 | 0.6644 |
| Patient (mean across scans) | 0.8857 | 0.7009 |

Patient-level official-test Macro-F1 = **0.6653**, AUROC = **0.7697**, AD recall = **0.4733**, NC recall = **0.8500**. Of 131 AD patients, 69 were classified as NC. **The minimum 0.80 test-accuracy target was not met at any level.** Whether the prescribed 0.80 refers to slices, scans or patients is awaiting tutor confirmation; report all three distinctly.

A simple CNN baseline was trained and reached a reported validation **slice-level** accuracy of approximately **0.7038** in an earlier run. Its **official-test** metrics and full same-split, same-protocol benchmark comparison have **not yet been assembled**. Do not compare this CNN validation figure with the ConvNeXt test figure as if they were the same metric.

A patient-balanced sampling experiment was completed, but it did not outperform the original model on validation. The MRI foreground-intensity normalization experiment remains **exploratory**; passing preprocessing checks is not evidence of improved accuracy.

### Compute cost (partial; complete before final submission)

Compact ConvNeXt has approximately 1.9 million trainable parameters. Inference observed on a P100 was approximately **0.809 ms per slice for forward pass only** at batch size 16 (excluding image loading and transfer). Training records include `torch.cuda.max_memory_allocated()` peak **tensor allocation**, which is **not** total process GPU VRAM usage. Populate a verified cost table for both CNN and ConvNeXt from your run logs and record actual batch sizes and measurement methods.

## 5. Open research dilemma: confidence, ambiguity and clinical risk

The central question is whether the more complex ConvNeXt identifies ambiguous MRI cases more reliably than the baseline, or whether it confidently makes potentially harmful mistakes. Current test AD sensitivity is low (47.33% patient-level), demonstrating a major false-negative risk. The raw softmax values are **not clinically calibrated probabilities**. `predict.py` computes an illustrative confidence 0.80 review-mask diagnostic, but this is **not** an empirically calibrated threshold and **must not** be presented as a validated operating policy.

**Still required:** on development data, examine the relationship between confidence and correctness, evaluate calibration (e.g., reliability bins/ECE or Brier score), describe an explicit specialist-review threshold determined without optimizing on the official test, and quantify accepted-case coverage/error. Compare the same measures with the CNN baseline.

### 3–5 real failure-case autopsies — TODO

Select three to five representative MRI prediction errors (include false negatives and false positives), create authorized de-identified illustrations only if allowed under ADNI/course data policies, and discuss what anatomical, contrast, or scanning factors plausibly contributed. **No individual failure-case images or clinical diagnoses have yet been reviewed; do not invent them.**

### Engineering recommendation (provisional)

**Do not deploy autonomously.** The observed patient-level AD false-negative rate is too high for a clinical triage tool. Preserve human review, investigate robust development-set validation, intensity shifts, and slice-context limitations before considering any real-world use. Comparisons with the CNN and calibrated abstention criteria must be completed before final advice.

## 6. One-page feasibility review (midpoint check-off)

This is a draft synopsis; integrate the exact **one-page feasibility review shown to the tutor** into the final README as a clearly labelled section.

1. **User need/acceptance criteria:** clinical-research triage of AD vs CN; leak-free patient splits; course target ≥0.80 official test accuracy; baseline comparison; per-class metrics and uncertainty handling; compute profiling under 16 GB P100.
2. **Model/course concepts:** compact from-scratch ConvNeXt vs simple CNN; modern convolutional blocks; 2D slice training and probability fusion for scan/patient analysis.
3. **Initial feasibility:** 279/70/331 non-overlapping train/validation/test patients; functional CNN/ConvNeXt training/inference; official-test accuracy below target, requiring further analysis.
4. **Risks/budget/fallback:** limited independent patients, validation–test drop and missed AD cases; 2-hour GPU sessions with resumable checkpoints; future patient-level cross-validation; CNN + mandatory specialist review as fallback.

## 7. Artificial Intelligence Usage Disclosure

ChatGPT assisted with outlining/refactoring Python training and evaluation code, debugging Rangpur/environment issues, building scripts for auditing leakage and distributions, and explaining possible interpretation of the results. The student remains responsible for understanding, testing and verifying each model decision and statistic. Verification done so far includes patient-split assertions, observed Rangpur training/checkpoint outputs, repeated validation-metric checks, and official-test evaluation. The **consistent-evaluation refactor** has undergone local synthetic end-to-end checks (CNN training and inference, patient-level metric equality, legacy weights loading, and deliberate split mismatch rejection); it **must be checked on actual Rangpur data** before full reproducibility can be claimed. Add any other AI tools (e.g., GitHub Copilot) actually used, and cite adapted third-party code if applicable. This section must truthfully reflect the student's actual work.

## 8. Submission checklist — unfinished

- [ ] Confirm with tutor whether the 0.80 test target is slice/scan/patient-level.
- [ ] Complete midpoint tutor check-off and insert the actual one-page review.
- [ ] Verify the consistent-evaluation revision on Rangpur validation data (the earlier packaging smoke test predates this change); generate training curve plots from a new controlled run.
- [ ] Benchmark CNN and ConvNeXt on matched splits and metrics (including confidence comparison).
- [ ] Document calibrated/reject-option decision and 3–5 **real** error cases.
- [ ] Finish GPU memory/parameters/latency comparison and show plots.
- [ ] Record exact `pip freeze`/versions and provide realistic reproduction instructions.
- [ ] Create meaningful *progressive* student-verified Git commits on `topic-recognition`.
- [ ] Open upstream Pull Request; provide clear description and later respond to code review.
- [ ] Export final README to PDF for Turnitin; **do not** commit patient MRI images, ADNI data, `*.pth` checkpoints, or private derived test records.

### Git branch and path

Fork target: `https://github.com/shakes76/PatternAnalysis-2026`, branch `topic-recognition`, and dedicated project folder `recognition/ConvNeXt_Bai_s4985919/`. Confirm that the actual fork has all files and commit history before opening the upstream PR. AI-assisted code should be reviewed and understood before committing.

## 9. What changed in this review candidate (2026-10-14)

- Training can now choose its best checkpoint using the **same level and threshold** as prediction (slice/scan/patient); defaults are slice-level macro-F1 with AD cutoff 0.50. The historical checkpoint and its historical threshold 0.67 are **not retroactively altered**.
- `predict.py --model cnn` supports the simple baseline using the same 3-level metrics; it retains `--model convnext` and raw state_dict support.
- New runs write `best_meta.json`; `predict.py` checks class mapping, data-split membership/order, MRI transform representation, scan-to-patient mapping fingerprint and model name. Old `best.pth` has no such metadata and prints a warning rather than inventing verification.
- `train.py --test-only` correctly passes the model architecture to `predict.py`; official-test evaluation still requires explicit confirmation and is **not** needed for this refactoring check.
- The exact 80% accuracy definition (slice/scan/patient) is not specified in the provided PDF; seek tutor confirmation. No newly claimed test improvements are made here.
