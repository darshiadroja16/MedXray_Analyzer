"""
Converts RSNA dataset from DICOM + CSV format into what YOLO needs:
plain images (png) + one label .txt file per image.

Run this first, before anything else, in your Kaggle notebook.
"""

import os
import cv2
import pydicom
import pandas as pd
from tqdm import tqdm

# ---- change these paths if your Kaggle input folder is named differently ----
DICOM_DIR = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge/stage_2_train_images/"
LABELS_CSV = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
OUT_IMG_DIR = "/kaggle/working/dataset/images/train/"
OUT_LBL_DIR = "/kaggle/working/dataset/labels/train/"

# RSNA images are always 1024x1024 - we need this to normalize the box coordinates
IMG_SIZE = 1024

# how many images to convert today - full dataset is ~25000, we use a smaller
# subset first so training finishes in time for the demo
NUM_IMAGES = 2000


def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LBL_DIR, exist_ok=True)

    labels = pd.read_csv(LABELS_CSV)
    patient_ids = labels["patientId"].unique()[:NUM_IMAGES]

    for pid in tqdm(patient_ids, desc="converting DICOM to YOLO format"):
        dcm_path = os.path.join(DICOM_DIR, pid + ".dcm")
        if not os.path.exists(dcm_path):
            continue  # skip if the file is missing for some reason

        # read the DICOM file and pull out the actual image pixels
        dicom_file = pydicom.dcmread(dcm_path)
        image = dicom_file.pixel_array
        cv2.imwrite(os.path.join(OUT_IMG_DIR, pid + ".png"), image)

        # a patient can have 0, 1, or more pneumonia boxes - collect all of them
        patient_rows = labels[labels["patientId"] == pid]
        label_lines = []

        for _, row in patient_rows.iterrows():
            if row["Target"] == 1:  # Target=1 means this row has a real pneumonia box
                x, y, w, h = row["x"], row["y"], row["width"], row["height"]

                # YOLO wants center-x, center-y, width, height - all as a
                # fraction (0 to 1) of the image size, not raw pixels
                x_center = (x + w / 2) / IMG_SIZE
                y_center = (y + h / 2) / IMG_SIZE
                w_norm = w / IMG_SIZE
                h_norm = h / IMG_SIZE

                # class 0 = pneumonia (only one class in this problem)
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # if there's no pneumonia, we still write an empty label file -
        # YOLO needs to see "no object" examples too, otherwise it can't
        # learn what a normal/healthy X-ray looks like
        label_path = os.path.join(OUT_LBL_DIR, pid + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    print(f"done. converted {len(patient_ids)} images")


if __name__ == "__main__":
    main()