"""
Step 2b of Part C.

Trains the LateFusionClassifier (DenseNet-121 + ClinicalBERT) to predict,
from an X-ray and optional clinical notes, the 14 pathologies.
- Leverages patient-level splitting to prevent leakage.
- Handles missing notes with fallback zero representation (via dataset and projection).
- Uses BCEWithLogitsLoss with pos_weight for class imbalance.
- Performs validation-based threshold optimization.
- Computes comprehensive test metrics.
- Track runs under MLflow with clear synthetic/real text markers.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import mlflow
from transformers import AutoTokenizer
import config
from dataset import get_datasets
from models import LateFusionClassifier
from eval_utils import calculate_metrics, optimize_thresholds

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_pos_weights(dataset):
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]["label"].numpy())
    labels = np.array(labels)
    
    pos_weights = []
    num_samples = len(labels)
    for c in range(config.NUM_CLASSES):
        pos = np.sum(labels[:, c])
        neg = num_samples - pos
        weight = neg / (pos + 1e-8)
        pos_weights.append(weight)
        
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)

def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        text_available = batch["text_available"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask, text_available)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_loss(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_available = batch["text_available"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images, input_ids, attention_mask, text_available)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def predict_probabilities(model, dataloader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_available = batch["text_available"].to(device)
            labels = batch["label"].numpy()
            
            logits = model(images, input_ids, attention_mask, text_available)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(labels)
            
    return np.vstack(all_targets), np.vstack(all_probs)

def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # 1. Setup MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("MedXray_Module_C_Multimodal_Fusion")
    
    # Initialize clinical text tokenizer
    print(f"Loading tokenizer: {config.TEXT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Get dataset splits
    train_set, val_set, test_set = get_datasets(config.LABELS_MULTILABEL_CSV, config.IMAGES_DIR, tokenizer)
    
    is_synthetic = len(train_set) < 100
    data_label = "SYNTHETIC VALIDATION DATA" if is_synthetic else "REAL MIMIC/NIH CLINICAL DATA"
    print(f"\n==============================================================")
    print(f" TRAINING MULTIMODAL FUSION ON: {data_label}")
    print(f"==============================================================\n")
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Model setup
    model = LateFusionClassifier(image_backbone="densenet121").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Class imbalance weights
    pos_weight = compute_pos_weights(train_set)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Local paths
    best_model_path = config.MODEL_DIR / "fusion_multilabel_best.pt"
    final_model_path = config.MODEL_DIR / "fusion_multilabel_final.pt"
    metadata_path = config.MODEL_DIR / "fusion_metadata.json"
    
    with mlflow.start_run(run_name="clinicalbert_densenet_late_fusion") as run:
        # Log training info
        mlflow.log_param("architecture", "LateFusionClassifier")
        mlflow.log_param("image_backbone", "densenet121")
        mlflow.log_param("text_model", config.TEXT_MODEL_NAME)
        mlflow.log_param("epochs", config.EPOCHS)
        mlflow.log_param("learning_rate", config.LEARNING_RATE)
        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("seed", config.SEED)
        # Clearly flag synthetic notes verification vs real clinical evaluations
        mlflow.log_param("dataset_type", "synthetic_notes_validation" if is_synthetic else "real_paired_data")
        
        best_val_loss = float("inf")
        
        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            val_loss = evaluate_loss(model, val_loader, loss_fn)
            
            print(f"Epoch {epoch + 1}/{config.EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to {best_model_path}")
                
        # Save final model state
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        # Load best model for evaluations
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # 3. Optimize thresholds on Val
        print("Optimizing thresholds on validation set...")
        val_targets, val_probs = predict_probabilities(model, val_loader)
        optimal_thresholds = optimize_thresholds(val_targets, val_probs)
        print("Optimal Thresholds:", optimal_thresholds)
        
        # 4. Evaluate on Test set
        print("Evaluating on test set...")
        test_targets, test_probs = predict_probabilities(model, test_loader)
        
        macro_opt, per_class_opt = calculate_metrics(test_targets, test_probs, thresholds=optimal_thresholds)
        
        print("\n--- TEST METRICS (Optimized Thresholds) ---")
        for k, v in macro_opt.items():
            print(f"Macro {k}: {v:.4f}")
            mlflow.log_metric(f"test_macro_{k.lower()}", v)
            
        # Log per-class metrics
        for c_idx, c_name in enumerate(config.PATHOLOGIES):
            class_metrics = per_class_opt[c_name]
            mlflow.log_param(f"threshold_{c_name.lower()}", optimal_thresholds[c_idx])
            for metric_name, val in class_metrics.items():
                if metric_name != "Threshold":
                    mlflow.log_metric(f"test_{c_name.lower()}_{metric_name.lower()}", val)
                    
        # Log final model file
        mlflow.log_artifact(str(best_model_path))
        
        # Save metadata config locally
        metadata = {
            "model_name": "LateFusionClassifier",
            "image_backbone": "densenet121",
            "text_model": config.TEXT_MODEL_NAME,
            "is_synthetic": is_synthetic,
            "seed": config.SEED,
            "class_names": config.PATHOLOGIES,
            "optimal_thresholds": optimal_thresholds,
            "image_size": config.IMAGE_SIZE,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "test_macro_metrics": macro_opt
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        mlflow.log_artifact(str(metadata_path))
        print("Metadata saved to", metadata_path)
        print("MLflow Multimodal run completed successfully!")

if __name__ == "__main__":
    main()
