import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

#paths
metadata_path = "path to csv metadata file"
image_dir = "path to input folder/whole dataset"
subset1_train_dir = "path to subset/train"
subset1_test_dir = "path to subset/train"

# create output folders
os.makedirs(subset1_train_dir, exist_ok=True)
os.makedirs(subset1_test_dir, exist_ok=True)
print("Created subset1_train and subset1_test directories.")

# load metadata
df = pd.read_csv(metadata_path)
print(f"Loaded metadata with {len(df)} entries.")

#check for unique values
print("🔍 Unique values in 'used_in_subset':", df['used_in_subset'].unique())
print("🔍 Unique values in 'subset_id':", df['subset_id'].unique())

#subset1
df['used_in_subset'] = df['used_in_subset'].astype(str).str.strip().str.lower()
df['subset_id'] = pd.to_numeric(df['subset_id'], errors='coerce')

subset1_df = df[(df['used_in_subset'].isin(['yes', 'true', '1'])) & (df['subset_id'] == 1.0)]

print(f"Found {len(subset1_df)} images in subset 1.")
if subset1_df.empty:
    raise ValueError("No images found for subset 1.")

# Split into train/test
train_df, test_df = train_test_split(
    subset1_df,
    test_size=0.20,
    stratify=subset1_df['source'],
    random_state=42
)
print(f" Split into {len(train_df)} train and {len(test_df)} test images.")


def copy_images(df_subset, dest_dir):
    for _, row in df_subset.iterrows():
        src = os.path.join(image_dir, row['filename'])
        dst = os.path.join(dest_dir, row['filename'])
        try:
            shutil.copy2(src, dst)
        except FileNotFoundError:
            print(f"File not found: {src}")


copy_images(train_df, subset1_train_dir)
print("✅ Train images copied.")

copy_images(test_df, subset1_test_dir)
print("✅ Test images copied.")

print("\nSubset 1 train-test created successfully!")
