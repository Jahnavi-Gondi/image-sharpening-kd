import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#config
metadata_path = "path to csv metadata file"
image_dir = "path to whole dataset/ input dataset"
subset_dir = "path to output folder to save images to"
subset_size = 450 #change size accordingly
subset_id = "9" #change accordingly

#create output dir
os.makedirs(subset_dir, exist_ok=True)
print("Created subset directory (if not already present).")

#load metadata
df = pd.read_csv(metadata_path)
print(f"Loaded metadata with {len(df)} entries.")

#filter available images
available_df = df[
    (df['used_in_subset'] == 'no') &
    (df['subset_id'].isna()) &
    (df['used_in_benchmark_dev'] != 'yes') &
    (df['used_in_benchmark_final'] != 'yes')
]
print(f"Found {len(available_df)} images available for Subset {subset_id}")

#sample by source column
if 'source' not in available_df.columns:
    raise KeyError("Column 'source' not found in metadata.")

subset_df, _ = train_test_split(
    available_df,
    train_size=subset_size,
    stratify=available_df['source'],
    random_state=42
)

#a check for overlapping
prev_used = df[df['subset_id'].isin(['1', '2', '3', '4', '5', '6', '7', '8'])]['filename'].values
overlap = set(prev_used) & set(subset_df['filename'].values)
if overlap:
    print(f"Warning: {len(overlap)} images overlap ")
else:
    print("No overlap detected.")

#copy images
for _, row in subset_df.iterrows():
    src = os.path.join(image_dir, row['filename'])
    dst = os.path.join(subset_dir, row['filename'])
    shutil.copy2(src, dst)
print("Image copied")

#update metadata
df.loc[subset_df.index, 'used_in_subset'] = 'yes'
df.loc[subset_df.index, 'subset_id'] = subset_id

df.to_csv(metadata_path, index=False)
print("Metadata updated with subset info.")


print("\nSubset creation complete!")
print(f"Subset ID: {subset_id}")
print(f"Images saved in: {subset_dir}")
