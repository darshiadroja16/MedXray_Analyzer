"""
Step 3 of Part C.

Grad-CAM answers "which part of the image made the model predict this pathology?"
by showing which region the model paid the most attention to.
Refactored to:
- Work with the modular ImageEncoder (for both Visual-only and Late Fusion models).
- Extract activations and gradients from the last conv layer dynamically.
- Implement proper medical disclaimers on explainability vs. correctness.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import config
from models import ImageOnlyClassifier, LateFusionClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_gradcam(model, image_path, pathology_name, metadata, text_inputs=None):
    """
    Generates a Grad-CAM heatmap for a given image and pathology class.
    Args:
        model: PyTorch model instance (ImageOnlyClassifier or LateFusionClassifier)
        image_path: Path to the input image file
        pathology_name: Name of the pathology class to explain
        metadata: Model metadata JSON containing class names and thresholds
        text_inputs: Dictionary containing 'input_ids', 'attention_mask', and 'text_available' (for fusion model)
    """
    model.eval()
    
    # 1. Determine target class index
    class_names = metadata.get("class_names", config.PATHOLOGIES)
    if pathology_name not in class_names:
        raise ValueError(f"Pathology {pathology_name} not found in class names: {class_names}")
    class_idx = class_names.index(pathology_name)
    
    # 2. Extract target conv layer for gradients (stored in ImageEncoder)
    if hasattr(model, "image_encoder"):
        # LateFusionClassifier and ImageOnlyClassifier both expose .image_encoder
        target_layer = model.image_encoder.last_conv_layer
    else:
        raise AttributeError(
            "Model does not have an 'image_encoder' attribute. "
            "Ensure you pass an ImageOnlyClassifier or LateFusionClassifier instance."
        )
        
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.update({"value": grad}))

    # Register hook
    handle_f = target_layer.register_forward_hook(save_activation)

    # 3. Preprocess image
    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    original_image = Image.open(image_path).convert("L")
    tensor = transform(original_image).unsqueeze(0).to(device)
    tensor.requires_grad_()

    # 4. Forward pass
    if text_inputs is not None:
        # Late Fusion prediction
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        text_available = text_inputs["text_available"].to(device)
        logits = model(tensor, input_ids, attention_mask, text_available)
    else:
        # Image-only baseline prediction
        logits = model(tensor)
        
    probs = torch.sigmoid(logits)
    score = logits[0, class_idx]

    # 5. Backward pass
    model.zero_grad()
    score.backward()

    # 6. Extract heatmap
    grads = gradients["value"]
    acts = activations["value"]
    
    # Remove hooks
    handle_f.remove()

    # Global average pooling of gradients
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    acts_map = acts.squeeze(0).clone()

    # Weighted sum of feature map activations
    for i in range(acts_map.shape[0]):
        acts_map[i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts_map, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU on heatmap
    heatmap /= (heatmap.max() + 1e-8)  # Normalize
    
    prob = float(probs[0, class_idx].item())
    
    return heatmap, prob

def save_overlay(image_path, heatmap, output_path):
    """
    Overlays the heatmap on top of the original image and saves it.
    """
    original = cv2.imread(str(image_path))
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(str(output_path), overlay)
    
    # Save raw heatmap
    raw_heatmap_path = str(output_path).replace(".png", "_raw.png")
    cv2.imwrite(raw_heatmap_path, np.uint8(255 * heatmap_resized))

if __name__ == "__main__":
    # Test script loading visual baseline and running Grad-CAM on mock image
    print("Testing Grad-CAM pipeline...")
    mock_img = config.IMAGES_DIR / "00000000_000.png"
    if not mock_img.exists():
        print("Mock image not found. Run generate_mock_data.py first.")
    else:
        metadata_path = config.MODEL_DIR / "densenet_metadata.json"
        model_path = config.MODEL_DIR / "densenet_multilabel_best.pt"
        
        if not model_path.exists():
            print("Checkpoint not found. Run train_classifier.py first.")
        else:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                
            model = ImageOnlyClassifier("densenet121").to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Generate Grad-CAM for Pneumonia
            heatmap, score = generate_gradcam(model, mock_img, "Pneumonia", meta)
            print(f"Pneumonia predicted prob: {score:.4f}")
            
            out_path = config.RESULTS_DIR / "gradcam_example_pneumonia.png"
            save_overlay(mock_img, heatmap, out_path)
            print(f"Saved Grad-CAM overlays to {config.RESULTS_DIR}")
