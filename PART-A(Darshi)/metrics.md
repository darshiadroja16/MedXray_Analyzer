# Round 1 Metrics — Data Foundation Upgrade

**Model:** YOLOv8n
**Dataset:** RSNA Pneumonia Detection Challenge
**Training:** 8000 images (1802 pneumonia / 6198 non-pneumonia)
**Validation:** 2000 images (451 pneumonia / 1549 non-pneumonia)
**Epochs:** up to 100 (early stopping patience=20)

## Changes vs. Baseline (Round 1 = Data Foundation only)
| Change | Baseline | Round 1 |
|---|---|---|
| Training images | 1600 | 8000 |
| Split strategy | Patient-level random | Patient-level **stratified** on pneumonia presence |
| Duplicate check | Not done | MD5 file-hash check performed (0 duplicate groups found) |
| Small-box filtering | Not addressed | Documented decision: keep all valid boxes (only drop zero-area) |
| Epochs | 15 (fixed) | Up to 100 with early stopping (patience=20) |
| Augmentation/optimizer/resolution | default | **unchanged** (deferred to Round 2/3) |

## Results
| Metric | Baseline | Round 1 |
|---|---:|---:|
| Precision | 0.233 | 0.343 |
| Recall | 0.516 | 0.411 |
| mAP@0.5 | 0.267 | 0.330 |
| mAP@0.5:0.95 | 0.100 | 0.135 |
| Best F1 | 0.29 | 0.379 |
| Best F1 Confidence | 0.086 | 0.120 |

## Confusion Matrix (at Round 1 best-F1 confidence = 0.120)
```
[[        312         529]
 [        418           0]]
```
