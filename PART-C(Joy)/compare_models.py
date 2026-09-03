"""
Step 2c of Part C.

A lightweight comparison and ablation framework.
Compares visual-only architectures:
- DenseNet-121
- ResNet-50
- EfficientNet-B0
And runs an ablation study:
- DenseNet-121 (Visual-only) vs DenseNet-121 + ClinicalBERT (Late Fusion)

Logs hyper-parameters and outputs comparison reports to PART-C(Joy)/results/
"""

import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import config
from dataset import get_datasets
from models import ImageOnlyClassifier, LateFusionClassifier
from eval_utils import calculate_metrics, optimize_thresholds

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, dataloader, optimizer, loss_fn, is_fusion=False):
    model.train()
    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if is_fusion:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_available = batch["text_available"].to(device)
            logits = model(images, input_ids, attention_mask, text_available)
        else:
            logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, is_fusion=False):
    model.eval()
    all_probs = []
    all_targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()
            if is_fusion:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                text_available = batch["text_available"].to(device)
                logits = model(images, input_ids, attention_mask, text_available)
            else:
                logits = model(images)
                
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels)
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_inference_time = total_time / len(dataloader.dataset)
    
    return np.vstack(all_targets), np.vstack(all_probs), avg_inference_time

def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    train_set, val_set, test_set = get_datasets(config.LABELS_MULTILABEL_CSV, config.IMAGES_DIR, tokenizer)
    
    is_synthetic = len(train_set) < 100
    data_label = "SYNTHETIC VALIDATION DATA" if is_synthetic else "REAL MEDICAL DATA"
    print(f"\n==============================================================")
    print(f" RUNNING COMPARISON & ABLATION STUDY ON: {data_label}")
    print(f"==============================================================\n")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Pre-calculate positive weights from training data labels
    labels_list = [train_set[i]["label"].numpy() for i in range(len(train_set))]
    train_labels = np.array(labels_list)
    pos_weights = []
    for c in range(config.NUM_CLASSES):
        pos = np.sum(train_labels[:, c])
        neg = len(train_labels) - pos
        pos_weights.append(neg / (pos + 1e-8))
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Models to evaluate
    configurations = [
        {"name": "DenseNet-121 (Visual-only)", "backbone": "densenet121", "is_fusion": False},
        {"name": "ResNet-50 (Visual-only)", "backbone": "resnet50", "is_fusion": False},
        {"name": "EfficientNet-B0 (Visual-only)", "backbone": "efficientnet_b0", "is_fusion": False},
        {"name": "Late Fusion (DenseNet-121 + ClinicalBERT)", "backbone": "densenet121", "is_fusion": True}
    ]
    
    results = []
    
    for config_item in configurations:
        print(f"\n--- Training configuration: {config_item['name']} ---")
        
        # Instantiate model
        if config_item["is_fusion"]:
            model = LateFusionClassifier(image_backbone=config_item["backbone"]).to(device)
        else:
            model = ImageOnlyClassifier(backbone_name=config_item["backbone"]).to(device)
            
        params = get_param_count(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Fast comparison: Train for 1 epoch
        print("Training for 1 fast comparison epoch...")
        train_one_epoch(model, train_loader, optimizer, loss_fn, is_fusion=config_item["is_fusion"])
        
        # Validation for optimal thresholds
        print("Optimizing thresholds on validation set...")
        val_t, val_p, _ = evaluate_model(model, val_loader, is_fusion=config_item["is_fusion"])
        opt_thresholds = optimize_thresholds(val_t, val_p)
        
        # Test predictions and timing
        print("Evaluating on test set...")
        test_t, test_p, inf_time = evaluate_model(model, test_loader, is_fusion=config_item["is_fusion"])
        
        # Calculate metrics using optimal thresholds
        macro_metrics, _ = calculate_metrics(test_t, test_p, thresholds=opt_thresholds)
        
        results.append({
            "Model": config_item["name"],
            "AUROC": f"{macro_metrics['AUROC']:.4f}",
            "F1": f"{macro_metrics['F1']:.4f}",
            "Precision": f"{macro_metrics['Precision']:.4f}",
            "Recall": f"{macro_metrics['Recall']:.4f}",
            "Parameters (M)": f"{params / 1e6:.2f}M",
            "Inference Time (ms)": f"{inf_time * 1000:.2f} ms"
        })
        
    # Generate markdown table
    df_results = pd.DataFrame(results)
    
    # Save results to markdown file
    output_md_path = config.RESULTS_DIR / "comparison_results.md"
    
    with open(output_md_path, "w") as f:
        f.write("# Model Comparison & Ablation Study Results\n\n")
        f.write(f"**Data Environment:** {data_label}\n")
        f.write("> [!NOTE]\n")
        f.write("> Training was run for 1 epoch per configuration to verify pipelines under CPU resource constraints.\n\n")
        f.write("### Benchmark Results\n\n")
        
        # Write manual markdown table
        headers = df_results.columns.tolist()
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for _, row in df_results.iterrows():
            f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
            
        f.write("\n\n### Ablation Study Summary\n")
        f.write("- **DenseNet-121 (Visual-only):** Serves as our primary baseline model extracting 2D visual patterns from the radiographs.\n")
        f.write("- **Late Fusion (DenseNet-121 + ClinicalBERT):** Integrates text representations. Under synthetic notes where diagnostic cues correlate, we expect this model to achieve significantly higher F1/AUROC, confirming that the fusion architecture correctly passes and utilizes textual signals.\n")
        
    print("\nBenchmark table generated:")
    print(df_results.to_string(index=False))
    print(f"\nSaved results to {output_md_path}")

if __name__ == "__main__":
    main()
