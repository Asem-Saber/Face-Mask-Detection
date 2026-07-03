"""
Dataset Preparation & Augmentation Script for YOLO
Converts PASCAL VOC XML to YOLO format, applies Albumentations augmentations, 
and splits the data into Train/Val/Test directories.
"""

import os
import glob
import shutil
import yaml
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as albu

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
BASE_DIR = 'dataset'
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations')
AUG_OUTPUT_DIR = os.path.join(BASE_DIR, 'aug_images')

IMG_SIZE = 416
RANDOM_SEED = 42

CLASS_INDEX = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
INDEX_CLASS = {v: k for k, v in CLASS_INDEX.items()}
N_CLASSES = len(CLASS_INDEX)

# Define Augmentation Pipeline
AUGMENTER = albu.Compose([
    albu.Resize(height=IMG_SIZE, width=IMG_SIZE), 
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomBrightnessContrast(p=0.2),
    albu.GaussNoise(p=0.2),
], bbox_params=albu.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.1,
    min_area=0
))

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def yolo_format(size: tuple, box: tuple) -> tuple:
    """Converts PASCAL VOC format (xmin, ymin, xmax, ymax) to YOLO format (cx, cy, w, h) normalized."""
    dw = 1. / size[0]
    dh = 1. / size[1]

    cx = (box[0] + box[2]) / 2.
    cy = (box[1] + box[3]) / 2.
    w = box[2] - box[0]
    h = box[3] - box[1]

    return (cx * dw, cy * dh, w * dw, h * dh)

def extract_annotations(annot_dir: str) -> pd.DataFrame:
    """Parses XML annotations and returns a formatted DataFrame."""
    annots_rows = []
    xml_files = sorted([f for f in os.listdir(annot_dir) if f.endswith('.xml')])

    for filename in tqdm(xml_files, desc="Parsing XML Annotations"): 
        file_path = os.path.join(annot_dir, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        for obj in root.findall('object'): 
            class_name = obj.find('name').text
            if class_name not in CLASS_INDEX:
                continue
                
            class_id = CLASS_INDEX[class_name]
            bbox = obj.find('bndbox')

            xmin = max(0.0, float(bbox.find('xmin').text))
            ymin = max(0.0, float(bbox.find('ymin').text))
            xmax = min(float(width), float(bbox.find('xmax').text))
            ymax = min(float(height), float(bbox.find('ymax').text))

            yolo_box = yolo_format((width, height), (xmin, ymin, xmax, ymax))

            annots_rows.append({
                'file': filename,
                "center_x": yolo_box[0],
                "center_y": yolo_box[1],
                "width": yolo_box[2],
                "height": yolo_box[3],
                "class_id": class_id,
                "class_name": class_name
            })

    return pd.DataFrame(annots_rows)

def apply_augmentation(row: pd.Series):
    """Applies albumentations transforms to a single image and its bounding boxes."""
    img_path = row['image_path']
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = np.stack([
        row['center_x'],
        row['center_y'],
        row['width'],
        row['height']
    ], axis=-1).astype(np.float32)

    labels = row['class_id']

    augmented = AUGMENTER(image=img, bboxes=bboxes, class_labels=labels)
    return augmented['image'], augmented['bboxes'], augmented['class_labels']

def augment_dataset(data: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Iterates through the dataset, applies augmentations, and saves the new images."""
    os.makedirs(output_dir, exist_ok=True)
    aug_rows = []

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Generating Augmentations"): 
        aug_img, aug_bboxes, aug_labels = apply_augmentation(row)

        if aug_img is None or len(aug_bboxes) == 0:
            continue

        filename = os.path.basename(row['image_path'])
        new_filename = f"aug_{idx}_{filename}"
        aug_img_path = os.path.join(output_dir, new_filename)
        
        cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        aug_rows.append({
            'image_path': aug_img_path,
            'center_x': [b[0] for b in aug_bboxes],
            'center_y': [b[1] for b in aug_bboxes],
            'width': [b[2] for b in aug_bboxes],
            'height': [b[3] for b in aug_bboxes],
            'class_id': aug_labels,
            'class_name': [INDEX_CLASS[l] for l in aug_labels]
        })

    return pd.DataFrame(aug_rows)

def prepare_yolo_directories():
    """Creates the necessary folder structure for YOLO training."""
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(BASE_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, split, 'labels'), exist_ok=True)

def copy_yolo_data(df: pd.DataFrame, split_name: str):
    """Copies images and generates YOLO format .txt label files for a specific split."""
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {split_name} split"):
        img_path = row['image_path']
        filename = os.path.basename(img_path)
        file_prefix = os.path.splitext(filename)[0]

        dest_img_path = os.path.join(BASE_DIR, split_name, 'images', filename)
        dest_label_path = os.path.join(BASE_DIR, split_name, 'labels', f"{file_prefix}.txt")

        if os.path.exists(img_path):
            shutil.copy(img_path, dest_img_path)
        else:
            print(f"Warning: Could not find image at {img_path}. Skipping.")
            continue

        with open(dest_label_path, "w") as f:
            bboxes = zip(row['class_id'], row['center_x'], row['center_y'], row['width'], row['height'])
            for cls_id, cx, cy, w, h in bboxes:
                f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def create_yaml_config():
    """Generates the data.yaml file required by YOLO."""
    yaml_data = {
        'path': os.path.abspath(BASE_DIR),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': N_CLASSES,
        'names': list(CLASS_INDEX.keys())
    }

    yaml_path = 'data.yaml'
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        print(f"[INFO] Successfully created {yaml_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create YAML file: {e}")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    print("[INFO] Starting Dataset Preparation...")
    
    # 1. Load Images and Annotations
    imgs_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    annots_df = extract_annotations(ANNOTATIONS_DIR)

    annots_df['base_name'] = annots_df['file'].apply(lambda x: os.path.splitext(x)[0])
    img_df = pd.DataFrame({'image_path': imgs_paths})
    img_df['base_name'] = img_df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # Merge and Group
    full_dataset = pd.merge(img_df, annots_df, on='base_name', how='inner')
    data = full_dataset.groupby('image_path').agg({
        'center_x': list,
        'center_y': list, 
        'width': list,
        'height': list,
        'class_id': list, 
        "class_name": list
    }).reset_index()

    # 2. Apply Augmentations
    print(f"[INFO] Found {len(data)} original images. Starting augmentations...")
    aug_df = augment_dataset(data, AUG_OUTPUT_DIR)
    
    # Combine original and augmented data
    final_data = pd.concat([data, aug_df], ignore_index=True)
    print(f"[INFO] Total images after augmentation: {len(final_data)}")

    # 3. Train/Val/Test Split
    train_set, test_set = train_test_split(final_data, test_size=0.1, random_state=RANDOM_SEED)
    train_set, val_set = train_test_split(train_set, train_size=0.7, random_state=RANDOM_SEED)

    # 4. Prepare Directories and Copy Files
    prepare_yolo_directories()
    copy_yolo_data(train_set, 'train')
    copy_yolo_data(val_set, 'valid')
    copy_yolo_data(test_set, 'test')

    # 5. Create Configuration File
    create_yaml_config()
    print("[INFO] Dataset preparation complete!")

if __name__ == '__main__':
    main()