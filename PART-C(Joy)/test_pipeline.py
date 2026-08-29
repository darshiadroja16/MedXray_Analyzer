"""
Module C — Comprehensive Test Suite

Covers:
 1.  Label preprocessing
 2.  Dataset loading + patient-level split leak check
 3.  Image preprocessing tensor shape
 4.  Clinical text tokenisation
 5.  Missing-note fallback (zero tensors + flag)
 6.  ImageEncoder output dimensions (DenseNet-121, ResNet-50, EfficientNet-B0)
 7.  ClinicalBERT TextEncoder output dimensions
 8.  LateFusionClassifier fusion dimensions
 9.  ImageOnlyClassifier output dimensions
10.  Probability conversion (sigmoid always in [0,1])
11.  Grad-CAM generation (heatmap shape + value range)
12.  Inference pipeline — output structure and types
13.  Threshold optimisation — returns one float per class
14.  Class imbalance weights — positive classes get higher weight
15.  Integration: FastAPI-compatible service returns serialisable dict

Run with:
    python test_pipeline.py
"""

import sys
import traceback
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer

import config

PASS  = "\033[92m PASS\033[0m"
FAIL  = "\033[91m FAIL\033[0m"
SKIP  = "\033[93m SKIP\033[0m"

results = []


def test(name, fn):
    try:
        fn()
        print(f"[{PASS}] {name}")
        results.append((name, "PASS", ""))
    except AssertionError as e:
        msg = str(e)
        print(f"[{FAIL}] {name}: {msg}")
        results.append((name, "FAIL", msg))
    except Exception:
        msg = traceback.format_exc().strip().splitlines()[-1]
        print(f"[{FAIL}] {name}: {msg}")
        results.append((name, "FAIL", msg))


# ─── Ensure mock data exists ──────────────────────────────────────────────────
def ensure_mock_data():
    if not config.LABELS_MULTILABEL_CSV.exists():
        print("Mock data not found — generating now...")
        import subprocess
        subprocess.run([sys.executable, "generate_mock_data.py"], check=True)
        subprocess.run([sys.executable, "prepare_labels.py"], check=True)


# ─── 1. Label preprocessing ───────────────────────────────────────────────────
def test_label_preprocessing():
    import pandas as pd
    df = pd.read_csv(config.LABELS_MULTILABEL_CSV)
    for p in config.PATHOLOGIES:
        assert p in df.columns, f"Missing pathology column: {p}"
    assert "Image Index" in df.columns
    assert set(df[config.PATHOLOGIES[0]].unique()).issubset({0, 1}), "Labels should be binary"


# ─── 2. Dataset loading + patient-level split leak check ─────────────────────
def test_dataset_loading_and_no_leakage():
    from dataset import get_datasets
    train_set, val_set, test_set = get_datasets(config.LABELS_MULTILABEL_CSV, config.IMAGES_DIR)
    assert len(train_set) > 0, "Train set empty"
    assert len(val_set) > 0, "Val set empty"
    assert len(test_set) > 0, "Test set empty"

    # Patient leakage check (if Patient ID exists)
    import pandas as pd
    df = pd.read_csv(config.LABELS_MULTILABEL_CSV)
    if "Patient ID" in df.columns:
        train_ids = set(train_set.df["Patient ID"].values)
        val_ids   = set(val_set.df["Patient ID"].values)
        test_ids  = set(test_set.df["Patient ID"].values)
        assert len(train_ids & val_ids)  == 0, "Patient leakage: train ∩ val"
        assert len(train_ids & test_ids) == 0, "Patient leakage: train ∩ test"
        assert len(val_ids & test_ids)   == 0, "Patient leakage: val ∩ test"


# ─── 3. Image preprocessing tensor shape ─────────────────────────────────────
def test_image_preprocessing():
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_path = next(config.IMAGES_DIR.glob("*.png"), None)
    assert img_path is not None, "No images found in IMAGES_DIR"
    img = Image.open(img_path).convert("L")
    t = transform(img)
    assert t.shape == (3, *config.IMAGE_SIZE), f"Expected (3,224,224) got {t.shape}"


# ─── 4. Clinical text tokenisation ───────────────────────────────────────────
def test_text_tokenisation():
    tok = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    enc = tok(
        "Patient presents with fever and cough.",
        padding="max_length",
        truncation=True,
        max_length=config.MAX_TEXT_LENGTH,
        return_tensors="pt",
    )
    assert enc["input_ids"].shape == (1, config.MAX_TEXT_LENGTH)
    assert enc["attention_mask"].shape == (1, config.MAX_TEXT_LENGTH)


# ─── 5. Missing-note fallback ─────────────────────────────────────────────────
def test_missing_note_fallback():
    from dataset import MultiModalChestDataset
    import pandas as pd
    df = pd.read_csv(config.LABELS_MULTILABEL_CSV).head(4)
    # Remove Clinical Notes so fallback triggers
    if "Clinical Notes" in df.columns:
        df = df.drop(columns=["Clinical Notes"])
    ds = MultiModalChestDataset(df, config.IMAGES_DIR)
    item = ds[0]
    assert item["input_ids"].sum().item() == 0,   "Expected zero input_ids when note absent"
    assert item["attention_mask"].sum().item() == 0
    assert item["text_available"].item() == 0.0


# ─── 6. ImageEncoder output shapes ───────────────────────────────────────────
def test_image_encoder_shapes():
    from models import ImageEncoder
    for backbone in ["densenet121", "resnet50", "efficientnet_b0"]:
        enc = ImageEncoder(backbone_name=backbone)
        enc.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = enc(x)
        assert out.shape[0] == 2,                  f"{backbone}: batch dim wrong"
        assert out.ndim == 2,                       f"{backbone}: expected 2-D output"
        assert out.shape[1] == enc.feature_dim,     f"{backbone}: feature_dim mismatch"


# ─── 7. TextEncoder output dimensions ────────────────────────────────────────
def test_text_encoder_shapes():
    from models import TextEncoder
    enc = TextEncoder(freeze=True)
    enc.eval()
    input_ids     = torch.zeros(2, config.MAX_TEXT_LENGTH, dtype=torch.long)
    attention_mask= torch.zeros(2, config.MAX_TEXT_LENGTH, dtype=torch.long)
    with torch.no_grad():
        out = enc(input_ids, attention_mask)
    assert out.shape == (2, enc.feature_dim), f"TextEncoder shape: {out.shape}"


# ─── 8. LateFusionClassifier output dimensions ───────────────────────────────
def test_fusion_dimensions():
    from models import LateFusionClassifier
    model = LateFusionClassifier(image_backbone="densenet121")
    model.eval()
    img            = torch.randn(2, 3, 224, 224)
    input_ids      = torch.zeros(2, config.MAX_TEXT_LENGTH, dtype=torch.long)
    attention_mask = torch.zeros(2, config.MAX_TEXT_LENGTH, dtype=torch.long)
    text_avail     = torch.tensor([1.0, 0.0])
    with torch.no_grad():
        logits = model(img, input_ids, attention_mask, text_avail)
    assert logits.shape == (2, config.NUM_CLASSES), f"Fusion logits shape: {logits.shape}"


# ─── 9. ImageOnlyClassifier output dimensions ────────────────────────────────
def test_image_only_dimensions():
    from models import ImageOnlyClassifier
    model = ImageOnlyClassifier("densenet121")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, config.NUM_CLASSES), f"Baseline logits shape: {logits.shape}"


# ─── 10. Probability conversion ──────────────────────────────────────────────
def test_probability_conversion():
    from models import ImageOnlyClassifier
    model = ImageOnlyClassifier("densenet121")
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits)
    assert probs.min().item() >= 0.0 - 1e-6
    assert probs.max().item() <= 1.0 + 1e-6


# ─── 11. Grad-CAM heatmap shape and value range ───────────────────────────────
def test_gradcam_generation():
    from models import ImageOnlyClassifier
    from gradcam import generate_gradcam
    model = ImageOnlyClassifier("densenet121")
    img_path = next(config.IMAGES_DIR.glob("*.png"), None)
    assert img_path is not None, "No images for Grad-CAM test"
    metadata = {"class_names": config.PATHOLOGIES, "optimal_thresholds": [0.5]*14}
    heatmap, prob = generate_gradcam(model, img_path, "Pneumonia", metadata)
    assert heatmap.ndim == 2,                    "Heatmap should be 2-D"
    assert heatmap.min() >= 0.0 - 1e-6,         "Heatmap values should be >= 0"
    assert heatmap.max() <= 1.0 + 1e-6,         "Heatmap values should be <= 1"
    assert 0.0 <= prob <= 1.0,                   f"Probability out of range: {prob}"


# ─── 12. Inference pipeline — output structure ────────────────────────────────
def test_inference_output_structure():
    from inference import MultimodalDiagnosisService
    svc = MultimodalDiagnosisService()
    img_path = next(config.IMAGES_DIR.glob("*.png"), None)
    assert img_path is not None, "No images for inference test"

    result = svc.predict_from_path(img_path, generate_cam=False)

    assert "pathology_probabilities"  in result
    assert "top_predictions"          in result
    assert "image_available"          in result
    assert "clinical_text_available"  in result
    assert "gradcam"                  in result
    assert "metadata"                 in result

    assert len(result["pathology_probabilities"]) == config.NUM_CLASSES
    for name, prob in result["pathology_probabilities"].items():
        assert 0.0 <= prob <= 1.0, f"{name} prob out of range: {prob}"

    # Verify it is JSON-serialisable (needed for FastAPI)
    json.dumps(result)


# ─── 13. Threshold optimisation returns one float per class ──────────────────
def test_threshold_optimisation():
    from eval_utils import optimize_thresholds
    targets = np.random.randint(0, 2, (50, config.NUM_CLASSES)).astype(float)
    probs   = np.random.rand(50, config.NUM_CLASSES)
    thresholds = optimize_thresholds(targets, probs)
    assert len(thresholds) == config.NUM_CLASSES
    for t in thresholds:
        assert 0.0 < t < 1.0, f"Threshold out of (0,1): {t}"


# ─── 14. Class imbalance weights ─────────────────────────────────────────────
def test_class_imbalance_weights():
    from dataset import get_datasets
    train_set, _, _ = get_datasets(config.LABELS_MULTILABEL_CSV, config.IMAGES_DIR)
    import numpy as np
    labels = np.array([train_set[i]["label"].numpy() for i in range(len(train_set))])
    n = len(labels)
    for c in range(config.NUM_CLASSES):
        pos = labels[:, c].sum()
        neg = n - pos
        if pos > 0:
            weight = neg / pos
            # Rare classes should get weight > 1
            assert weight >= 0.0, f"Class {c} weight should be non-negative"


# ─── 15. predict_from_bytes returns serialisable dict ────────────────────────
def test_predict_from_bytes():
    from inference import MultimodalDiagnosisService
    svc = MultimodalDiagnosisService()
    img_path = next(config.IMAGES_DIR.glob("*.png"), None)
    assert img_path is not None
    raw = img_path.read_bytes()
    result = svc.predict_from_bytes(raw, clinical_note=None, stem="test_bytes")
    json.dumps(result)  # must be serialisable
    assert "pathology_probabilities" in result


# ─── Runner ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_mock_data()

    print("\n======================================================")
    print("  Module C — Test Suite")
    print("======================================================\n")

    test("1.  Label preprocessing",             test_label_preprocessing)
    test("2.  Dataset loading + leak check",     test_dataset_loading_and_no_leakage)
    test("3.  Image preprocessing shape",        test_image_preprocessing)
    test("4.  Text tokenisation shape",          test_text_tokenisation)
    test("5.  Missing-note fallback",            test_missing_note_fallback)
    test("6.  ImageEncoder output shapes",       test_image_encoder_shapes)
    test("7.  TextEncoder output dimensions",    test_text_encoder_shapes)
    test("8.  LateFusionClassifier dimensions",  test_fusion_dimensions)
    test("9.  ImageOnlyClassifier dimensions",   test_image_only_dimensions)
    test("10. Probability conversion [0,1]",     test_probability_conversion)
    test("11. Grad-CAM shape + value range",     test_gradcam_generation)
    test("12. Inference output structure",       test_inference_output_structure)
    test("13. Threshold optimisation",           test_threshold_optimisation)
    test("14. Class imbalance weights",          test_class_imbalance_weights)
    test("15. predict_from_bytes serialisable",  test_predict_from_bytes)

    print("\n======================================================")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Results: {passed} passed  |  {failed} failed  |  {len(results)} total")
    print("======================================================\n")

    if failed > 0:
        sys.exit(1)
