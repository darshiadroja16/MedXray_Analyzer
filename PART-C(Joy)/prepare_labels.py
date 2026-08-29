"""
Step 1 of Part C.

The NIH ChestX-ray14 dataset gives labels as a single text column, e.g.
"Cardiomegaly|Effusion" or "No Finding". This script turns that into a
proper multi-label format: one column per pathology, 1 if present, 0 if
not - which is what a PyTorch dataloader needs to train a classifier.
It preserves extra metadata columns like 'Patient ID' and 'Clinical Notes' if available.
"""

import pandas as pd
import config

def main():
    print(f"Reading raw CSV from: {config.RAW_CSV}")
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {config.RAW_CSV}. "
            "Please run 'python generate_mock_data.py' first to set up mock data for testing."
        )

    df = pd.read_csv(config.RAW_CSV)

    print("Mapping label columns...")
    for pathology in config.PATHOLOGIES:
        # 1 if this pathology's name appears in the "Finding Labels" text, else 0
        df[pathology] = df["Finding Labels"].apply(lambda labels: int(pathology in str(labels)))

    # Keep Patient ID and Clinical Notes if they are present in raw data
    extra_cols = []
    if "Patient ID" in df.columns:
        extra_cols.append("Patient ID")
    if "Clinical Notes" in df.columns:
        extra_cols.append("Clinical Notes")

    output_columns = ["Image Index"] + extra_cols + config.PATHOLOGIES
    df[output_columns].to_csv(config.LABELS_MULTILABEL_CSV, index=False)

    print(f"Done. Wrote {len(df)} rows to {config.LABELS_MULTILABEL_CSV}")

if __name__ == "__main__":
    main()
