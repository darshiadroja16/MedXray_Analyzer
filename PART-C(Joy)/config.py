import os
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "Data_Entry_2017.csv"
LABELS_MULTILABEL_CSV = DATA_DIR / "labels_multilabel.csv"
IMAGES_DIR = DATA_DIR / "images"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True, parents=True)

# Training Hyper-parameters
SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3          # Kept low for CPU resource constraints; adjust for production
LEARNING_RATE = 1e-4

# Pathologies list
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(PATHOLOGIES)

# ClinicalBERT parameters
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_TEXT_LENGTH = 128
FREEZE_TEXT_ENCODER = True

# MLflow config — use SQLite backend (file-store was deprecated in MLflow ≥3.0)
# The DB is created automatically on first run inside the project directory.
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

# Kaggle overrides
if "KAGGLE_CONTAINER_NAME" in os.environ or Path("/kaggle").exists():
    IMAGES_DIR = Path("/kaggle/input/nih-chest-xrays/sample/images")
    RAW_CSV = Path("/kaggle/input/nih-chest-xrays/data/Data_Entry_2017.csv")
