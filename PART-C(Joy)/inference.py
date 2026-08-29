"""
Module C — Inference Service (FastAPI-compatible)

Exposes MultimodalDiagnosisService, a self-contained class that Part D
(FastAPI backend) can import and call without knowing internal details.

Usage from FastAPI:
    from inference import MultimodalDiagnosisService
    service = MultimodalDiagnosisService()          # loads model once at startup
    result = service.predict_from_path("xray.png")  # image-only
    result = service.predict_from_path("xray.png", clinical_note="Patient has cough...")
    result = service.predict_from_bytes(image_bytes, clinical_note="...")

Output (JSON-serialisable dict):
{
    "pathology_probabilities":  {"Atelectasis": 0.12, ...},
    "top_predictions":          [{"pathology": "Pneumonia", "probability": 0.82}, ...],
    "image_available":          true,
    "clinical_text_available":  false,
    "gradcam": {
        "Pneumonia": "/absolute/path/to/gradcam_Pneumonia.png"
    },
    "metadata": {
        "model_name": "...",
        "is_synthetic_run": false,
        "thresholds_used": {...}
    }
}
"""

import io
import json
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer

import config
from models import ImageOnlyClassifier, LateFusionClassifier
from gradcam import generate_gradcam, save_overlay

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalDiagnosisService:
    """
    Integration-ready multi-modal diagnosis service for the MedXray platform.

    Load once at application startup; reuse for every request.
    Designed to be imported directly into a FastAPI app:

        from inference import MultimodalDiagnosisService
        service = MultimodalDiagnosisService()

    The service auto-detects whether a fusion checkpoint exists; if not it
    falls back to the image-only baseline gracefully.
    """

    def __init__(
        self,
        fusion_checkpoint: Optional[Path] = None,
        baseline_checkpoint: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        gradcam_output_dir: Optional[Path] = None,
        top_k: int = 5,
    ):
        fusion_checkpoint    = fusion_checkpoint    or config.MODEL_DIR / "fusion_multilabel_best.pt"
        baseline_checkpoint  = baseline_checkpoint  or config.MODEL_DIR / "densenet_multilabel_best.pt"
        metadata_path        = metadata_path        or config.MODEL_DIR / "densenet_metadata.json"
        self.gradcam_dir     = gradcam_output_dir   or config.RESULTS_DIR / "gradcam"
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        self.top_k           = top_k

        # ── Load metadata (class names, thresholds) ──────────────────────────
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            warnings.warn(
                f"Metadata not found at {metadata_path}. "
                "Using default PATHOLOGIES list and 0.5 thresholds."
            )
            self.metadata = {
                "class_names": config.PATHOLOGIES,
                "optimal_thresholds": [0.5] * config.NUM_CLASSES,
                "is_synthetic": True,
            }

        self.class_names   = self.metadata["class_names"]
        self.thresholds    = self.metadata.get("optimal_thresholds", [0.5] * config.NUM_CLASSES)
        self.is_synthetic  = self.metadata.get("is_synthetic", True)

        # ── Load tokenizer ────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # ── Load model: prefer fusion, fall back to baseline ─────────────────
        if fusion_checkpoint.exists():
            self.model = LateFusionClassifier(image_backbone="densenet121").to(device)
            self.model.load_state_dict(
                torch.load(fusion_checkpoint, map_location=device)
            )
            self.use_fusion   = True
            self.model_label  = "LateFusionClassifier (DenseNet-121 + ClinicalBERT)"
            print(f"Loaded fusion model from {fusion_checkpoint}")
        elif baseline_checkpoint.exists():
            self.model = ImageOnlyClassifier(backbone_name="densenet121").to(device)
            self.model.load_state_dict(
                torch.load(baseline_checkpoint, map_location=device)
            )
            self.use_fusion   = False
            self.model_label  = "ImageOnlyClassifier (DenseNet-121 baseline)"
            print(f"Loaded baseline model from {baseline_checkpoint}")
        else:
            warnings.warn(
                "No trained checkpoint found. Running with random weights for pipeline validation only."
            )
            self.model       = ImageOnlyClassifier(backbone_name="densenet121").to(device)
            self.use_fusion  = False
            self.model_label = "ImageOnlyClassifier (untrained — validation mode)"

        self.model.eval()

        # ── Image transform (matches training pre-processing) ─────────────────
        self.transform = T.Compose([
            T.Resize(config.IMAGE_SIZE),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        img = Image.open(image_path).convert("L")
        return self.transform(img).unsqueeze(0).to(device)

    def _encode_note(self, note: Optional[str]):
        """Returns (input_ids, attention_mask, text_available flag)."""
        if note and note.strip():
            enc = self.tokenizer(
                note.strip(),
                padding="max_length",
                truncation=True,
                max_length=config.MAX_TEXT_LENGTH,
                return_tensors="pt",
            )
            return (
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                torch.tensor([1.0], device=device),
            )
        return (
            torch.zeros(1, config.MAX_TEXT_LENGTH, dtype=torch.long, device=device),
            torch.zeros(1, config.MAX_TEXT_LENGTH, dtype=torch.long, device=device),
            torch.tensor([0.0], device=device),
        )

    def _run_inference(self, image_tensor, input_ids, attention_mask, text_available):
        """Returns 14 probabilities as a numpy array."""
        with torch.no_grad():
            if self.use_fusion:
                logits = self.model(image_tensor, input_ids, attention_mask, text_available)
            else:
                logits = self.model(image_tensor)
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()

    def _build_result(
        self,
        probs: np.ndarray,
        image_path: Path,
        clinical_note: Optional[str],
        text_flag: float,
        gradcam_paths: dict,
    ) -> dict:
        prob_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        thresholds_used = {name: float(self.thresholds[i]) for i, name in enumerate(self.class_names)}

        # Top-K predictions above threshold
        top_preds = sorted(
            [
                {"pathology": name, "probability": float(probs[i])}
                for i, name in enumerate(self.class_names)
                if probs[i] >= self.thresholds[i]
            ],
            key=lambda x: x["probability"],
            reverse=True,
        )[: self.top_k]

        return {
            "pathology_probabilities":  prob_dict,
            "top_predictions":          top_preds,
            "image_available":          True,
            "clinical_text_available":  bool(text_flag > 0.5),
            "gradcam":                  gradcam_paths,
            "metadata": {
                "model_name":       self.model_label,
                "is_synthetic_run": self.is_synthetic,
                "thresholds_used":  thresholds_used,
            },
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_from_path(
        self,
        image_path,
        clinical_note: Optional[str] = None,
        generate_cam: bool = True,
        cam_pathologies: Optional[list] = None,
    ) -> dict:
        """
        Predict pathologies from an X-ray file path.

        Args:
            image_path: str or Path to the chest X-ray image.
            clinical_note: Optional free-text clinical note.
            generate_cam: Whether to produce Grad-CAM heatmaps.
            cam_pathologies: List of pathology names to visualise (default: top predictions).

        Returns:
            JSON-serialisable result dict.
        """
        image_path = Path(image_path)
        image_tensor = self._load_image_tensor(image_path)
        input_ids, attention_mask, text_flag = self._encode_note(clinical_note)

        probs = self._run_inference(image_tensor, input_ids, attention_mask, text_flag)

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        gradcam_paths = {}
        if generate_cam:
            # Default: top-3 predicted classes
            if cam_pathologies is None:
                sorted_idx = np.argsort(probs)[::-1][:3]
                cam_pathologies = [self.class_names[i] for i in sorted_idx]

            for pathology in cam_pathologies:
                try:
                    text_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "text_available": text_flag,
                    } if self.use_fusion else None

                    heatmap, _ = generate_gradcam(
                        self.model, image_path, pathology, self.metadata, text_inputs
                    )
                    out_path = self.gradcam_dir / f"gradcam_{pathology}_{image_path.stem}.png"
                    save_overlay(image_path, heatmap, out_path)
                    gradcam_paths[pathology] = str(out_path)
                except Exception as e:
                    gradcam_paths[pathology] = f"error: {e}"

        return self._build_result(probs, image_path, clinical_note, float(text_flag.item()), gradcam_paths)

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        clinical_note: Optional[str] = None,
        stem: str = "uploaded",
    ) -> dict:
        """
        Predict pathologies from raw image bytes (e.g., FastAPI UploadFile).

        Args:
            image_bytes: Raw bytes of the image file.
            clinical_note: Optional free-text clinical note.
            stem: Filename stem used for Grad-CAM output naming.

        Returns:
            JSON-serialisable result dict (no Grad-CAM — save from bytes requires a temp file).
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        image_tensor = self.transform(img).unsqueeze(0).to(device)

        input_ids, attention_mask, text_flag = self._encode_note(clinical_note)
        probs = self._run_inference(image_tensor, input_ids, attention_mask, text_flag)

        # Grad-CAM requires a saved file path; save temp and run
        tmp_path = config.RESULTS_DIR / f"_tmp_{stem}.png"
        img.save(tmp_path)
        gradcam_paths = {}
        sorted_idx = np.argsort(probs)[::-1][:3]
        for i in sorted_idx:
            pathology = self.class_names[i]
            try:
                text_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "text_available": text_flag,
                } if self.use_fusion else None
                heatmap, _ = generate_gradcam(
                    self.model, tmp_path, pathology, self.metadata, text_inputs
                )
                out_path = self.gradcam_dir / f"gradcam_{pathology}_{stem}.png"
                save_overlay(tmp_path, heatmap, out_path)
                gradcam_paths[pathology] = str(out_path)
            except Exception as e:
                gradcam_paths[pathology] = f"error: {e}"
        tmp_path.unlink(missing_ok=True)

        return self._build_result(probs, Path(stem), clinical_note, float(text_flag.item()), gradcam_paths)


# ── Stand-alone smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("=== Module C — Inference Service smoke test ===\n")
    service = MultimodalDiagnosisService()

    example_img = config.IMAGES_DIR / "00000000_000.png"
    if not example_img.exists():
        print("Mock images not found — run generate_mock_data.py first.")
    else:
        # Image-only prediction
        result_img_only = service.predict_from_path(example_img)
        print("--- Image-only prediction ---")
        pprint.pprint(result_img_only)

        # Image + clinical note
        result_multimodal = service.predict_from_path(
            example_img,
            clinical_note="Patient presents with cough and fever. Findings consistent with pneumonia.",
        )
        print("\n--- Multimodal prediction (image + note) ---")
        pprint.pprint(result_multimodal)
