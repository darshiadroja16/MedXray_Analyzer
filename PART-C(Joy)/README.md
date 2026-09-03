# Part C — Multi-Modal Medical Image Analysis (Module C)

**Owner:** Joy  
**Status:** Complete — image-only baseline trained; multimodal pipeline implemented and validated on synthetic data; awaiting real paired clinical-note data for full multimodal evaluation.

---

## Module Objective

Predict the probability of **14 thoracic pathologies** from a chest X-ray, optionally supplemented by clinical notes, and produce Grad-CAM heatmaps that explain which image regions drove each prediction.

---

## Architecture

```
                ┌─────────────────────────┐
X-ray ──────────│  DenseNet-121 Encoder   │──► Image Embedding (1024-d)
                └─────────────────────────┘         │
                                                     ▼
                                              Image Projection (256-d)
                                                     │
                                                     ├──────────► Concatenate ──► Fusion MLP ──► 14 logits ──► sigmoid ──► probabilities
                                                     │
                ┌─────────────────────────┐         ▼
Clinical Notes ─│  ClinicalBERT Encoder   │──► Text Embedding (768-d)
                └─────────────────────────┘         │
                                              Text Projection (256-d)
```

**Why Late Fusion?**  
Each modality is processed independently through its domain-specialist encoder before any information is shared. This means:
- Grad-CAM operates cleanly on the visual encoder's convolutional activations.
- The image branch can run standalone when notes are absent.
- Each encoder can be debugged, replaced, or fine-tuned independently.

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal chest X-ray images labelled with 14 pathologies (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia).

> **Clinical Notes Status:** The NIH ChestX-ray14 dataset does **not** include free-text radiology reports. No legitimate paired image-text dataset is present in this repository. The multi-modal training pipeline is ready to run immediately once a paired dataset (e.g., MIMIC-CXR) is provided. See **[Clinical Note Limitation](#clinical-note-data-limitation)** below.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | Centralised configuration — paths, hyperparameters, model names, MLflow URI |
| `generate_mock_data.py` | Generates synthetic X-rays + labels + notes **for pipeline validation only** |
| `prepare_labels.py` | Converts NIH raw CSV into binary multi-label matrix |
| `dataset.py` | `MultiModalChestDataset` — patient-level splitting, tokenisation, missing-note fallback |
| `models.py` | `ImageEncoder`, `TextEncoder`, `LateFusionClassifier`, `ImageOnlyClassifier` |
| `eval_utils.py` | Shared metrics (AUROC, F1, Sensitivity, Specificity, PR-AUC) + threshold optimisation |
| `train_classifier.py` | Trains image-only DenseNet-121 baseline with MLflow logging |
| `train_fusion.py` | Trains DenseNet-121 + ClinicalBERT late-fusion model with MLflow logging |
| `compare_models.py` | Benchmarks DenseNet-121 / ResNet-50 / EfficientNet-B0; ablation: image-only vs multimodal |
| `gradcam.py` | Disease-specific Grad-CAM heatmaps; works with both baseline and fusion models |
| `inference.py` | `MultimodalDiagnosisService` — FastAPI-compatible, importable by Part D |
| `test_pipeline.py` | 15-test suite covering every pipeline stage |

---

## DenseNet-121 Justification

| Model | Params | CheXpert Mean AUC | Notes |
|-------|--------|-------------------|-------|
| VGG-16 | 138 M | 0.852 | Too heavy for deployment |
| ResNet-50 | 25 M | 0.881 | Strong baseline |
| **DenseNet-121** | **8 M** | **0.898** | **Best AUC, fewest parameters** |
| EfficientNet-B0 | 5 M | 0.895 | Good, but slightly lower AUC |

DenseNet-121's dense connections maximise gradient flow, reduce vanishing gradients, and encourage feature reuse — all beneficial for the subtle texture differences in chest radiographs. The original CheXNet paper (Rajpurkar et al., 2017) used exactly this architecture to achieve radiologist-level performance.

---

## ClinicalBERT Justification

**Checkpoint:** `emilyalsentzer/Bio_ClinicalBERT` (Alsentzer et al., 2019, NAACL)  
Pre-trained on MIMIC-III clinical notes — the same domain as radiology reports. Generic BERT lacks medical vocabulary and fails to resolve clinical abbreviations (e.g., "SOB" = shortness of breath).

---

## Multi-Label Classification Justification

A patient can have **multiple pathologies simultaneously** (e.g., Effusion + Atelectasis). Softmax forces a single-class prediction; **Sigmoid** treats each pathology as an independent binary decision, which is medically correct. Loss: `BCEWithLogitsLoss` with class-weighted `pos_weight` to handle the severe label imbalance in the NIH dataset.

---

## Grad-CAM Explanation

Grad-CAM computes the gradient of the target class score with respect to the final convolutional feature map, then weights each feature channel by its global average gradient. The result is a coarse spatial heatmap highlighting which image regions most influenced the prediction.

> **Important disclaimer:** A Grad-CAM heatmap visualises *model sensitivity*, not clinical correctness. It is an interpretability tool, not a radiological diagnosis.

---

## Training Instructions

```bash
# 1. Set up mock data (skip if you have real NIH data)
python generate_mock_data.py
python prepare_labels.py

# 2. Train image-only baseline (logs to MLflow)
python train_classifier.py

# 3. Train multimodal late-fusion model (logs to MLflow)
#    ONLY run on real paired data for clinical evaluation.
#    Synthetic notes are for software integration verification only.
python train_fusion.py

# 4. View MLflow experiments
mlflow ui --backend-store-uri mlruns
```

For the real NIH dataset, replace `data/images/` and `data/Data_Entry_2017.csv` with the downloaded dataset and re-run `prepare_labels.py`.

---

## Evaluation Instructions

```bash
# Run model comparison and ablation study
python compare_models.py

# Results saved to results/comparison_results.md
```

---

## Inference Instructions

### Command line
```bash
python inference.py   # smoke-tests against the first mock image
```

### From Part D FastAPI backend
```python
from inference import MultimodalDiagnosisService

# Load once at startup
service = MultimodalDiagnosisService()

# Image-only prediction
result = service.predict_from_path("patient_xray.png")

# Image + clinical note
result = service.predict_from_path(
    "patient_xray.png",
    clinical_note="Patient presents with productive cough and fever for 3 days."
)

# From uploaded bytes (FastAPI UploadFile)
result = service.predict_from_bytes(await file.read(), clinical_note=note_text)
```

### Response structure
```json
{
  "pathology_probabilities": {"Pneumonia": 0.82, "Atelectasis": 0.21, ...},
  "top_predictions": [{"pathology": "Pneumonia", "probability": 0.82}],
  "image_available": true,
  "clinical_text_available": false,
  "gradcam": {"Pneumonia": "/path/to/gradcam_Pneumonia.png"},
  "metadata": {
    "model_name": "LateFusionClassifier (DenseNet-121 + ClinicalBERT)",
    "is_synthetic_run": false,
    "thresholds_used": {"Pneumonia": 0.42, ...}
  }
}
```

---

## Tests

```bash
python test_pipeline.py
```

Covers 15 tests: label preprocessing, patient-level split leak check, image shape, tokenisation, missing-note fallback, all encoder shapes, fusion dimensions, probability conversion, Grad-CAM, inference structure, threshold optimisation, class-imbalance weights, and FastAPI byte-input serialisability.

---

## Experiment Tracking

MLflow is used to track all training runs.

```bash
mlflow ui --backend-store-uri PART-C(Joy)/mlruns
```

Each run logs: backbone, epochs, LR, batch size, seed, dataset type (synthetic/real), train/val loss per epoch, test macro AUROC/F1/Precision/Recall/Sensitivity/Specificity/PR-AUC, per-class metrics, and optimal thresholds.

---

## Model Comparison

See [`results/comparison_results.md`](results/comparison_results.md) for the generated benchmark table (architecture comparison + image-only vs multimodal ablation study).

---

## Clinical Note Data Limitation

> **Status: Blocked — real paired data not available**

The NIH ChestX-ray14 dataset contains only pathology labels, not radiology reports. The multimodal pipeline has been fully implemented and validated with synthetic clinical notes that deliberately contain pathology-correlated keywords.

**Synthetic notes are used exclusively for:**
- Software integration testing
- Pipeline shape/dimension verification
- Confirming that the fusion architecture correctly passes textual signals

**Synthetic notes are never used for:**
- Reporting clinical performance metrics
- Claiming multimodal improvement over the image-only baseline

**To enable real multimodal training:**
1. Obtain a paired dataset — MIMIC-CXR (Johnson et al., 2019) links radiology reports to chest X-rays and is publicly available via PhysioNet.
2. Map reports to NIH image filenames via patient/study IDs.
3. Place reports in the `Clinical Notes` column of `Data_Entry_2017.csv`.
4. Re-run `prepare_labels.py` → `train_fusion.py`.

---

## Limitations

1. **No real paired clinical notes** — multimodal training metrics are synthetic-only.
2. **CPU-only environment** — model comparison uses 1-epoch quick runs; production training requires GPU.
3. **NIH label noise** — labels were extracted by NLP from reports; some are imprecise.
4. **Grad-CAM resolution** — heatmaps are coarse (7×7 feature map upsampled); not a substitute for proper lesion segmentation.
5. **Calibration** — probabilities are not temperature-calibrated; raw outputs should be treated as ranking scores, not clinical likelihoods.

---

## Future Work

- Train on MIMIC-CXR with real paired reports.
- Add temperature calibration post-training.
- Explore cross-attention fusion (early/intermediate) as an ablation.
- Replace Grad-CAM with GradCAM++ or RISE for higher-resolution heatmaps.
- Share DenseNet-121 weights with Part B (similar-case retrieval) to avoid duplicated computation.
- Add domain adaptation for deployment on hospital-specific data distributions.
