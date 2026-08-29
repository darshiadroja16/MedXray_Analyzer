# Implementation Plan - Multi-Modal Medical Image Analysis Platform (Module C)

Complete the implementation of Module C (Multi-Modal Diagnosis System) to support thoracic pathology classification from chest X-rays and clinical notes using DenseNet-121, ClinicalBERT, Late Fusion, and Grad-CAM explainability.

## User Review Required

> [!IMPORTANT]
> **No Legitimate Paired Clinical Notes in Repository:**
> Our audit confirmed that the repository does not contain paired chest X-rays and clinical notes. The NIH ChestX-ray14 dataset used by Part B and Part C only contains pathology labels (`Data_Entry_2017.csv`) and no free-text clinical reports.
> To address this, we will implement a synthetic text generation mode **strictly for pipeline validation, software integration, and debugging**. It will **never** be used for clinical performance evaluation. The real training pipeline is fully prepared to run immediately on real paired clinical note data.

> [!IMPORTANT]
> **Ablation Study & Model Comparison:**
> We will compare:
> 1. CNN Backbones (DenseNet-121 vs ResNet-50 vs EfficientNet-B0) on visual-only classification.
> 2. Visual-only baseline vs. Multi-modal Late Fusion (DenseNet-121 + ClinicalBERT) to evaluate the impact of text integration (ablation study).
> 
> Due to resource constraints (CPU environment), experiments will run with a quick, low-epoch training protocol.

> [!IMPORTANT]
> **Experiment Tracking with MLflow:**
> We will configure MLflow to log hyper-parameters, training metrics (loss per epoch), evaluation results (AUROC, F1, Sensitivity, etc.), optimal thresholds, and model checkpoints.

## Open Questions

No open questions remain. We are proceeding with the approved architectural changes.

---

## Proposed Changes

### 1. Configuration & Data Setup

#### [NEW] [config.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/config.py)
* Centralized configuration for all paths, training hyperparameters (batch size, learning rate, epochs), random seeds, MLflow settings, model configurations, and clinical note tokenization.

#### [NEW] [generate_mock_data.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/generate_mock_data.py)
* Generates synthetic chest X-rays, mock labels CSV with patient IDs, and synthetic clinical reports containing keywords correlating to specific pathologies.
* **Warning flags:** All synthetic files and output labels will explicitly indicate they are synthetic/demo data.

#### [MODIFY] [prepare_labels.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/prepare_labels.py)
* Prepares multi-label classification CSV from raw NIH files, reading paths from `config.py`.

---

### 2. Dataset & Models Architecture

#### [NEW] [dataset.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/dataset.py)
* Implements `MultiModalChestDataset` with:
  * Patient-level train/validation/test split to prevent leakage.
  * Image resizing, grayscale normalization, and ClinicalBERT tokenization.
  * Missing clinical note handling: uses a fallback zero-tensor representation.

#### [NEW] [models.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/models.py)
* Modular PyTorch architectures:
  * `ImageEncoder`: Configurable CNN feature extractor (DenseNet-121, ResNet-50, or EfficientNet-B0).
  * `TextEncoder`: ClinicalBERT encoder (`emilyalsentzer/Bio_ClinicalBERT`) with options to freeze weights.
  * `LateFusionClassifier`: Visual and text embeddings projected to compatible dimensions, concatenated, and classified using an MLP head.
  * `ImageOnlyClassifier`: Baseline classifier wrapping `ImageEncoder`.

---

### 3. Training, Comparison & Tracking

#### [MODIFY] [train_classifier.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/train_classifier.py)
* Trains the image-only baseline. Uses `BCEWithLogitsLoss` with class-weighted loss (`pos_weight`) to handle class imbalance.
* Performs validation-based threshold optimization (grid search for optimal F1-score threshold per class).
* Logs hyper-parameters, metrics (AUROC, F1, Precision, Recall, Sensitivity, Specificity, PR-AUC), optimal thresholds, and model states to MLflow.

#### [NEW] [train_fusion.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/train_fusion.py)
* Trains the `LateFusionClassifier` on multimodal data.
* Implements threshold optimization and MLflow logging.
* Explicitly logs metadata marking the run as "Synthetic Data Validation" or "Real Data Training".

#### [NEW] [compare_models.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/compare_models.py)
* Benchmarks visual architectures (DenseNet-121, ResNet-50, EfficientNet-B0) and visual-only vs multimodal late-fusion models (ablation study).
* Logs metrics and generates a final comparison table:
  Model | AUROC | F1 | Precision | Recall | Parameters | Inference Time | Modality

---

### 4. Explainability & FastAPI Integration

#### [MODIFY] [gradcam.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/gradcam.py)
* Refactored to map with the updated modular `ImageEncoder`.
* Outputs overlaid heatmaps for target diseases with clear interpretability disclaimers.

#### [NEW] [inference.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/inference.py)
* Implements an integration-ready backend service:
  * Contains a loadable helper class `MultimodalDiagnosisService` containing methods like `predict(image_bytes, clinical_note)` or `predict_from_path(image_path, clinical_note)`.
  * Returns structured JSON responses compatible with FastAPI endpoints (probabilities, top predictions, Grad-CAM heatmap paths, availability indicators, thresholds used).

#### [NEW] [test_pipeline.py](file:///c:/Users/Joy%20Patel/OneDrive/Desktop/Bootcamp/MedXray_Analyzer/MedXray_Analyzer.worktrees/analyze-partc-joy-folder/PART-C(Joy)/test_pipeline.py)
* Tests: data splitting patient leakage, text fallback logic, shape consistency across layers, late fusion dimensions, threshold tuning, and FastAPI-ready inference inputs/outputs.

---

## Verification Plan

### Automated Tests
* Run the unit and integration test suite:
  ```bash
  python test_pipeline.py
  ```

### Manual Verification
1. Run mock generation:
   ```bash
   python generate_mock_data.py
   ```
2. Prepare labels:
   ```bash
   python prepare_labels.py
   ```
3. Run visual baseline training with MLflow tracking:
   ```bash
   python train_classifier.py
   ```
4. Run late fusion training with MLflow tracking:
   ```bash
   python train_fusion.py
   ```
5. Run comparison and ablation framework:
   ```bash
   python compare_models.py
   ```
6. Run the FastAPI backend service simulation:
   ```bash
   python inference.py
   ```
