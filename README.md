# Multi-Modal Medical Image Analysis Platform

Builds a web platform that accepts chest X-rays (optional
clinical notes) and produces a pneumonia diagnosis with probability, region
segmentation, and a structured radiology report.

Dataset: RSNA Pneumonia Detection Challenge (Kaggle) + NIH ChestX-ray14.

## Team and parts

| Part | Owner  | What it does                                                   | Status                  |
|------|--------|------------------------------------------------------------------|--------------------------|
| A    | Darshi | YOLOv8 pneumonia bounding box detection                          | working baseline         |
| B    | Shreya | DenseNet embeddings + FAISS similar-case search                  | working baseline         |
| C    | Joy    | Multi-modal fusion (DenseNet + ClinicalBERT) + Grad-CAM           | partial baseline        |
| D    | Ankit  | Report generation, audit dashboard, deployment                   | not started              |

Each part runs independently on the same raw X-ray input. Part D is the only
place where all outputs come together.


Each part's folder has its own README with details specific to that part.
