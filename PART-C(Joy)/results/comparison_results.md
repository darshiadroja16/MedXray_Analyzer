# Model Comparison & Ablation Study Results

**Data Environment:** SYNTHETIC VALIDATION DATA
> [!NOTE]
> Training was run for 1 epoch per configuration to verify pipelines under CPU resource constraints.

### Benchmark Results

| Model | AUROC | F1 | Precision | Recall | Parameters (M) | Inference Time (ms) |
|---|---|---|---|---|---|---|
| DenseNet-121 (Visual-only) | 0.5208 | 0.1171 | 0.0714 | 0.3571 | 6.97M | 69.44 ms |
| ResNet-50 (Visual-only) | 0.4938 | 0.1171 | 0.0714 | 0.3571 | 23.54M | 92.45 ms |
| EfficientNet-B0 (Visual-only) | 0.5167 | 0.1310 | 0.0816 | 0.3571 | 4.03M | 72.79 ms |
| Late Fusion (DenseNet-121 + ClinicalBERT) | 0.5292 | 0.1667 | 0.1020 | 0.5000 | 7.48M | 186.00 ms |


### Ablation Study Summary
- **DenseNet-121 (Visual-only):** Serves as our primary baseline model extracting 2D visual patterns from the radiographs.
- **Late Fusion (DenseNet-121 + ClinicalBERT):** Integrates text representations. Under synthetic notes where diagnostic cues correlate, we expect this model to achieve significantly higher F1/AUROC, confirming that the fusion architecture correctly passes and utilizes textual signals.
