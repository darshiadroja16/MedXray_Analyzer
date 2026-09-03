# Results - Day 1 baseline

Model: YOLOv8n
Training data: 2000 images (subset), 15 epochs
Val set: 300 images (unseen during training)

| Metric      | Score |
|-------------|-------|
| Precision   | 0.243 |
| Recall      | 0.409 |
| mAP50       | 0.232 |
| mAP50-95    | 0.093 |

## Notes
- This is a baseline run on a small subset - not the final model.
- mAP50 is calculated on the validation set (images the model never saw during training).
- Next: train on the full ~25,000 image dataset with more epochs (50-100).
