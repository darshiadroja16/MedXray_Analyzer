# Task A — Pneumonia Detection with YOLOv8

Part of the **Multi-Modal Medical Image Analysis Platform** final year project.
This module trains a **YOLOv8** object detector on the **RSNA Pneumonia Detection
Challenge** dataset to localize lung-opacity (pneumonia) regions on frontal chest
X-rays.

---

## Repository Structure

```
Multi-Modal_Medical_Image_Analysis_Platform/
├── .dvc/                     # DVC internal config (remote, cache pointers)
├── .venv/                    # local virtual environment (not tracked)
└── PART-A(...)/
    ├── weights/
    │   ├── .gitignore        # keeps best.pt out of git
    │   ├── best.pt           # trained weights (pulled via DVC, not stored in git)
    │   └── best.pt.dvc       # DVC pointer file (tracked in git)
    ├── dataset.yaml           # YOLO dataset config (paths + class names)
    ├── metrics.md              # baseline training metrics
    ├── notebook.ipynb          # full training notebook (run on Kaggle)
    └── README.md                # this file
```

> **Why DVC?** `best.pt` is a binary model file — too large/inefficient to store
> directly in git. Git tracks the small `best.pt.dvc` pointer file, while the
> actual weight file lives in DVC remote storage and is pulled on demand.

---

## Requirements

- Python 3.9+
- Git
- [DVC](https://dvc.org/) (`pip install dvc`) — plus the DVC plugin for whichever
  remote is configured for this repo (e.g. `dvc[gdrive]`, `dvc[s3]`)
- A Kaggle account (only needed if training from scratch)

Core Python packages used across the notebook and inference:
   ultralytics pydicom opencv-python pandas numpy scikit-learn tqdm dvc

---

## 1. Clone the Repo

```bash
git clone <your-repo-url>
cd Multi-Modal_Medical_Image_Analysis_Platform
```

Set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install ultralytics pydicom opencv-python pandas numpy scikit-learn tqdm dvc
```

---

## Option 1 — Just Run Inference with the Pretrained Weights

Use this path if you don't want to retrain anything and just want predictions
from the already-trained model.

**Step 1 — Pull the weights via DVC**

```bash
cd "PART-A(...)"
dvc pull weights/best.pt.dvc
```

This downloads the actual `best.pt` file from the configured DVC remote into
`weights/best.pt`. (If `dvc pull` fails with a remote/auth error, make sure
you have access to the DVC remote this repo points to — check `.dvc/config`.)

**Step 2 — Run detection on an X-ray**

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(
    source="path/to/your_xray.png",   # or a folder of images
    conf=0.086,                        # best-F1 confidence from metrics.md
    save=True                          # saves annotated output image
)

results[0].show()   # display prediction with bounding boxes
```

Annotated outputs are saved under `runs/detect/predict/`.

---

## Option 2 — Train the Model from Scratch

Use this path if you want to reproduce or improve on the baseline.

### Step 1 — Get the RSNA dataset from Kaggle

1. Create a free account at [kaggle.com](https://www.kaggle.com) and join the
   competition: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
   (you must click **"Join Competition"** and accept the rules before you can
   download anything).
2. Get an API token: **Kaggle profile → Settings → API → Create New Token**.
   This downloads `kaggle.json`.
3. **Recommended:** skip local downloading entirely — open a new
   [Kaggle Notebook](https://www.kaggle.com/code), click **Add Input →
   Competitions**, search "RSNA Pneumonia Detection Challenge", and attach it.
   The data will be available at `/kaggle/input/rsna-pneumonia-detection-challenge/`
   with no download step needed. This is exactly how `notebook.ipynb` in this
   repo was run.
4. **If you want it locally instead:**
   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle competitions download -c rsna-pneumonia-detection-challenge
   unzip rsna-pneumonia-detection-challenge.zip -d rsna_data
   unzip rsna_data/stage_2_train_images.zip -d rsna_data/stage_2_train_images
   ```

### Step 2 — Convert DICOM + CSV labels into YOLO format

Open and run `notebook.ipynb` (in Kaggle, or locally after adjusting paths).
It performs the full pipeline:

1. Loads `stage_2_train_labels.csv` (`patientId, x, y, width, height, Target`).
2. Selects the first `NUM_IMAGES = 2000` unique patients and splits them
   **by `patientId`** into train/val (80/20, `random_state=42`) — this avoids
   the same patient's images leaking across the split.
3. For every patient: reads the `.dcm` file, normalizes pixel values to
   0–255, converts grayscale → 3-channel, resizes to `1024×1024`, and saves
   as PNG under `images/train/` or `images/val/`.
4. Converts each pneumonia box from pixel coordinates `(x, y, width, height)`
   into YOLO's normalized `(x_center, y_center, w, h)` format, one `.txt`
   label file per image (single class, id `0 = pneumonia`; empty file = no
   pneumonia present).
5. Writes `dataset.yaml` describing the resulting folder layout.

Resulting structure (matches what `dataset.yaml` expects):

```
rsna_yolo/
├── images/
│   ├── train/   (1600 PNGs)
│   └── val/     (400 PNGs)
└── labels/
    ├── train/   (1600 .txt files, one per image)
    └── val/     (400 .txt files)
```

> If you run this outside Kaggle, update the `path:` field in
> `dataset.yaml` to point at your local `rsna_yolo/` folder — the checked-in
> version points at the Kaggle working directory.

### Step 3 — Train YOLOv8

```bash
pip install ultralytics

yolo detect train \
  data=dataset.yaml \
  model=yolov8n.pt \
  epochs=15 \
  imgsz=1024 \
  patience=10
```

Trained weights will appear at `runs/detect/train/weights/best.pt`. Copy that
into `weights/best.pt` and re-run `dvc add weights/best.pt` + `dvc push` to
version the new weights (see below).

### Step 4 — Evaluate

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

Compare the output metrics against the current baseline in `metrics.md`.

---

## Current Baseline (see `metrics.md`)

**Model:** YOLOv8n · **Train:** 1600 images · **Val:** 400 images · **Epochs:** 15

| Metric | Value |
|---|---:|
| Precision | 0.233 |
| Recall | 0.516 |
| mAP@0.5 | 0.267 |
| mAP@0.5:0.95 | 0.100 |
| Best F1 | 0.29 |
| Best F1 Confidence | 0.086 |

**Confusion Matrix:** TP: 0 · FP: 0 · FN: 198 · Background: 198

This is an early baseline (small subset of data, only 15 epochs, nano model
size) — it's meant as a reference point for future experiments (more images,
more epochs, larger YOLOv8 variant, class-imbalance handling), not a final
result.

---

## Updating the Tracked Weights (DVC)

If you retrain and get a new `best.pt`:

```bash
dvc add weights/best.pt
git add weights/best.pt.dvc weights/.gitignore
git commit -m "Update baseline weights"
dvc push
git push
```

This keeps the large binary out of git history while still versioning it
alongside the code.
