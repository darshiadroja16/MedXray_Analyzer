import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
import config

class MultiModalChestDataset(Dataset):
    def __init__(self, df, images_dir, tokenizer=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Default transform matching ImageNet pre-training expects 3 channels
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(config.IMAGE_SIZE),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.images_dir / img_name
        
        # Load image (handle grayscale conversion and errors)
        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            # Fallback to zero tensor if image is corrupt or missing
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("L", config.IMAGE_SIZE)
            
        image_tensor = self.transform(image)
        
        # Check if clinical note is available
        note_available = 1.0
        note_text = ""
        
        if "Clinical Notes" in row and pd.notna(row["Clinical Notes"]):
            note_text = str(row["Clinical Notes"]).strip()
            if not note_text:
                note_available = 0.0
        else:
            note_available = 0.0
            
        # Tokenize or create empty text representations
        if self.tokenizer is not None and note_available == 1.0:
            encoding = self.tokenizer(
                note_text,
                padding="max_length",
                truncation=True,
                max_length=config.MAX_TEXT_LENGTH,
                return_tensors="pt"
            )
            # encoding is a dictionary, extract tensors and squeeze the batch dim
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            # Zero representation placeholder when text is missing
            input_ids = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            attention_mask = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            note_available = 0.0

        # Labels tensor
        labels = torch.tensor(row[config.PATHOLOGIES].values.astype("float32"))
        
        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_available": torch.tensor(note_available, dtype=torch.float32),
            "label": labels,
            "img_name": img_name
        }

def get_datasets(csv_path, images_dir, tokenizer=None, split_ratios=(0.7, 0.15, 0.15), seed=config.SEED):
    """
    Split labels CSV at patient-level to prevent data leakage and returns train, val, and test datasets.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV labels not found at {csv_path}. Run prepare_labels.py first.")
        
    df = pd.read_csv(csv_path)
    
    # 1. Split logic
    if "Patient ID" in df.columns:
        # Group by Patient ID to ensure patient-level separation
        unique_patients = df["Patient ID"].unique()
        
        # Reproducible shuffle
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_patients)
        
        n_patients = len(unique_patients)
        train_end = int(split_ratios[0] * n_patients)
        val_end = train_end + int(split_ratios[1] * n_patients)
        
        train_patients = unique_patients[:train_end]
        val_patients = unique_patients[train_end:val_end]
        test_patients = unique_patients[val_end:]
        
        train_df = df[df["Patient ID"].isin(train_patients)]
        val_df = df[df["Patient ID"].isin(val_patients)]
        test_df = df[df["Patient ID"].isin(test_patients)]
        
        print(f"Split by patient ID: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients.")
    else:
        # Fallback to simple random split if Patient ID is not available
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df_shuffled)
        train_end = int(split_ratios[0] * n)
        val_end = train_end + int(split_ratios[1] * n)
        
        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:]
        print("Patient ID not found. Fallback to row-level split.")
        
    print(f"Data splits generated: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
    
    # Check if there is data leakage
    if "Patient ID" in df.columns:
        train_set = set(train_df["Patient ID"])
        val_set = set(val_df["Patient ID"])
        test_set = set(test_df["Patient ID"])
        assert len(train_set.intersection(val_set)) == 0, "Leakage detected between train and val splits!"
        assert len(train_set.intersection(test_set)) == 0, "Leakage detected between train and test splits!"
        assert len(val_set.intersection(test_set)) == 0, "Leakage detected between val and test splits!"
        print("Verification passed: No patient leakage between splits.")

    train_dataset = MultiModalChestDataset(train_df, images_dir, tokenizer)
    val_dataset = MultiModalChestDataset(val_df, images_dir, tokenizer)
    test_dataset = MultiModalChestDataset(test_df, images_dir, tokenizer)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Test script loading dataset
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    try:
        train_set, val_set, test_set = get_datasets(config.LABELS_MULTILABEL_CSV, config.IMAGES_DIR, tokenizer)
        item = train_set[0]
        print("Success! First item shapes:")
        print("Image shape:", item["image"].shape)
        print("Input IDs shape:", item["input_ids"].shape)
        print("Label shape:", item["label"].shape)
        print("Text available flag:", item["text_available"].item())
    except Exception as e:
        print("Failed to run test:", e)
