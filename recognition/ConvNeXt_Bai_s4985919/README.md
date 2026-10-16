
# Alzheimer's Disease Classification Using ConvNeXt

COMP3710 Pattern Recognition and Analysis
The University of Queensland

## Project Overview

This project investigates Alzheimer's disease (AD)
classification against cognitively normal (CN) cases
using the ADNI brain MRI dataset.

A simple CNN is implemented as the baseline model.
A ConvNeXt model will be developed and evaluated
against this baseline.

## Current Progress

- ADNI dataset loading implemented.
- Training and test image counts checked.
- Image-ID overlap checks completed.
- Simple CNN baseline implemented.
- Initial training smoke test completed.
- ### Failure Case Analysis: ConvNeXt on ADNI MRI

To investigate the limitations of the proposed ConvNeXt model, five misclassified MRI slices were selected from the patient-independent validation set.

At the fixed AD probability threshold of 0.67, the model correctly classified 2,065 of 2,620 validation slices, achieving an accuracy of 78.82%. The remaining 555 errors comprised 245 false negatives (AD predicted as NC) and 310 false positives (NC predicted as AD).

Five cases from different patients were selected to illustrate both types of classification errors across different model scores.

| Case | Ground Truth | Prediction | P(AD) | Error Type |
|---|---|---|---|---|
| 1 | AD | NC | 0.240 | False Negative |
| 2 | AD | NC | 0.471 | False Negative |
| 3 | AD | NC | 0.634 | False Negative |
| 4 | NC | AD | 0.740 | False Positive |
| 5 | NC | AD | 0.949 | False Positive |

**Case 1 — False Negative (P(AD) = 0.240)**

The model assigns a relatively low AD score to an image labelled AD. This demonstrates a classification error that is not simply caused by the decision threshold being close to the predicted score. The underlying cause cannot be established from the image and model output alone.

**Case 2 — False Negative (P(AD) = 0.471)**

The model again fails to identify an AD-labelled slice. Its score differs substantially from Case 1, demonstrating that false negatives occur across multiple score ranges. Further anatomical or image-quality assessment would be needed to identify contributing factors.

**Case 3 — Threshold-Sensitive False Negative (P(AD) = 0.634)**

The AD probability is slightly below the predetermined threshold of 0.67. Consequently, the image is classified as NC despite having an AD score above 0.50. This illustrates how an operating threshold can affect individual classification decisions. It does not justify changing the threshold based on this example.

**Case 4 — False Positive (P(AD) = 0.740)**

This NC-labelled slice is incorrectly classified as AD, with a score moderately above the decision threshold. The image also appears relatively dark, but there is insufficient evidence to establish image brightness as the cause of the error.

**Case 5 — High-Score False Positive (P(AD) = 0.949)**

The model assigns a high AD probability to an NC-labelled image. This is a particularly important failure for the project's confidence and ambiguity investigation, because it demonstrates that a strong model score does not guarantee a correct classification.

However, the reported softmax score is not a calibrated clinical probability. A single example cannot establish the overall calibration quality of the model.

**Discussion and Engineering Implications**

These examples demonstrate two relevant limitations: false negatives can occur even when the model assigns low AD scores, and false positives can occur at high AD scores.

Visual differences in image brightness, contrast and visible anatomical structure may warrant further investigation, but no causal relationship has been established from these five images.

The cases were intentionally selected across different prediction scores and should not be interpreted as a representative sample of all validation errors.

The findings support the need for a more systematic confidence–correctness assessment before proposing any automated clinical triage policy. In particular, high prediction confidence alone should not be treated as sufficient evidence for safe autonomous decision-making.

All examples are drawn from the validation set rather than the official test set. No medical diagnosis has been inferred from the displayed slices.

## Planned Work

- Verify patient-level data separation.
- Train and evaluate the CNN baseline.
- Implement and train ConvNeXt.
- Compare classification performance and confidence.
- Analyse failure cases and computational costs.
