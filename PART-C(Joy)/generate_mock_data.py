import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import config

def generate_mock_images_and_csv(num_samples=40):
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} mock images at {config.IMAGES_DIR}...")
    
    # Pathologies mapping to notes keywords for synthetic signal learning
    pathology_keywords = {
        "Cardiomegaly": ["cardiomegaly", "enlarged heart shadow", "cardiac silhouette is enlarged"],
        "Pneumonia": ["pneumonia", "consolidation", "focal opacity", "airspace disease"],
        "Effusion": ["pleural effusion", "fluid in pleural cavity", "blunting of costophrenic angle"],
        "Infiltration": ["infiltration", "interstitial opacity", "patchy infiltrates"],
        "Atelectasis": ["atelectasis", "lobar collapse", "volume loss"],
        "Nodule": ["nodule", "round opacity", "small pulmonary nodule"],
        "Mass": ["mass", "large pulmonary mass", "soft tissue density"],
        "Pneumothorax": ["pneumothorax", "pleural line", "collapsed lung"],
        "Consolidation": ["consolidation", "lobar consolidation"],
        "Edema": ["edema", "pulmonary edema", "vascular congestion", "cephalization"],
        "Emphysema": ["emphysema", "hyperinflation", "flattened diaphragms"],
        "Fibrosis": ["fibrosis", "scarring", "fibrotic changes"],
        "Pleural_Thickening": ["pleural thickening", "pleural scarring"],
        "Hernia": ["hernia", "hiatal hernia", "diaphragmatic hernia"]
    }

    image_indices = []
    finding_labels_list = []
    patient_ids = []
    clinical_notes = []

    # Ensure some duplicate patient IDs to test patient-level separation and leak prevention
    patients = [f"{i:05d}" for i in range(1, num_samples // 2 + 1)]

    for i in range(num_samples):
        img_name = f"{i:08d}_000.png"
        img_path = config.IMAGES_DIR / img_name
        
        # 1. Create a dummy chest X-ray image (grayscale with lung-like shapes)
        img = Image.new("L", (224, 224), color=20)
        draw = ImageDraw.Draw(img)
        # Left lung ellipse
        draw.ellipse([30, 40, 95, 190], fill=60)
        # Right lung ellipse
        draw.ellipse([125, 40, 190, 190], fill=60)
        # Heart shape outline / fill (sometimes larger if cardiomegaly)
        has_cardiomegaly = random.random() < 0.2
        heart_width = 45 if has_cardiomegaly else 30
        draw.ellipse([112 - heart_width, 110, 112 + heart_width, 170], fill=120)
        
        # Save mock image
        img.save(img_path)
        
        # 2. Generate random finding labels
        active_pathologies = []
        if has_cardiomegaly:
            active_pathologies.append("Cardiomegaly")
        
        # Add random other pathologies
        for path, keywords in pathology_keywords.items():
            if path == "Cardiomegaly":
                continue
            if random.random() < 0.15:
                active_pathologies.append(path)
                
        if not active_pathologies:
            finding_labels = "No Finding"
        else:
            finding_labels = "|".join(active_pathologies)
            
        # 3. Generate patient ID
        patient_id = random.choice(patients)
        
        # 4. Generate corresponding clinical notes containing keywords (for fusion validation)
        note_sentences = []
        if finding_labels == "No Finding":
            note_sentences.append("Lungs are clear. No acute cardiopulmonary findings.")
        else:
            for path in active_pathologies:
                note_sentences.append(f"Findings are suggestive of {random.choice(pathology_keywords[path])}.")
            
            # Sometimes add irrelevant medical jargon
            note_sentences.append("The trachea is midline.")
            note_sentences.append("Bones and soft tissues are unremarkable.")
            
        random.shuffle(note_sentences)
        clinical_note = " ".join(note_sentences)
        
        image_indices.append(img_name)
        finding_labels_list.append(finding_labels)
        patient_ids.append(patient_id)
        clinical_notes.append(clinical_note)

    # Save mock raw labels CSV
    df = pd.DataFrame({
        "Image Index": image_indices,
        "Finding Labels": finding_labels_list,
        "Patient ID": patient_ids,
        "Clinical Notes": clinical_notes,
        "Patient Age": [random.randint(18, 90) for _ in range(num_samples)],
        "Patient Gender": [random.choice(["M", "F"]) for _ in range(num_samples)],
        "View Position": [random.choice(["AP", "PA"]) for _ in range(num_samples)]
    })
    
    df.to_csv(config.RAW_CSV, index=False)
    print(f"Mock labels and clinical notes saved to {config.RAW_CSV}")
    print(f"Dataset marked as: SYNTHETIC DATA FOR VALIDATION AND INTEGRATION ONLY.")

if __name__ == "__main__":
    generate_mock_images_and_csv()
