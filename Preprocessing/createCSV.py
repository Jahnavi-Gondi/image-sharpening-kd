import os
import pandas as pd
from PIL import Image

DATASET_DIR = "input dataset path"
OUTPUT_CSV = "path to csv metadata file to be created"


def infer_category(filename):
    fname = filename.lower()
    if 'text' in fname:
        return 'text'
    elif 'people' in fname or 'face' in fname:
        return 'people'
    elif 'nature' in fname or 'landscape' in fname:
        return 'nature'
    elif 'game' in fname:
        return 'games'
    else:
        return 'misc'


# Collect image files
image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Initialize metadata list
metadata = []

for fname in sorted(image_files):
    full_path = os.path.join(DATASET_DIR, fname)

    try:
        with Image.open(full_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"⚠️ Skipping {fname} due to error: {e}")
        width, height = 0, 0

    # Extract source prefix
    source = fname.split('_')[0].lower()
    category = infer_category(fname)

    metadata.append({
        'filename': fname,
        'source': source,
        'width': width,
        'height': height,
        'category': category,
        'used_in_subset': 'no',
        'subset_id': '',
        'used_in_benchmark_dev': 'no',
        'used_in_benchmark_final': 'no',
        'notes': ''
    })


df = pd.DataFrame(metadata)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Created {OUTPUT_CSV} with {len(df)} entries.")
