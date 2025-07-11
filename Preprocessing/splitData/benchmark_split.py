import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

#Paths
metadata_path = "path to csv metadata file"
image_dir = "path to whole dataset"
benchmark_dev_dir = "path to copy benchmark dev used for intermediate validation images to"
benchmark_final_dir = "path to benchmark final evaluation images folder"

#creating benchmark folders only if not exists
os.makedirs(benchmark_dev_dir, exist_ok=True)
os.makedirs(benchmark_final_dir, exist_ok=True)
print("Created benchmark directories (if they didn't already exist).")

#Load metadata
print("Loading metadata...")
df = pd.read_csv(metadata_path)
print(f"Loaded metadata with {len(df)} entries.")

#Filter out images already used in benchmarks
available_df = df[
    (df['used_in_benchmark_dev'].fillna("no") != "yes") &
    (df['used_in_benchmark_final'].fillna("no") != "yes")
]
print(f"Found {len(available_df)} available images for benchmark selection.")

#10% sample for benchmark
benchmark_df, _ = train_test_split(
    available_df,
    test_size=(1 - 0.10),  # 10% split
    stratify=available_df['source'],
    random_state=42
)
print(f"Selected {len(benchmark_df)} images for benchmark set.")

#Split benchmark 50/50 into dev and final
benchmark_dev_df, benchmark_final_df = train_test_split(
    benchmark_df,
    test_size=0.5,
    stratify=benchmark_df['source'],
    random_state=42
)
print(f"Split into {len(benchmark_dev_df)} for benchmark_dev and {len(benchmark_final_df)} for benchmark_final.")

#Copy images to respective folders
def copy_images(subset_df, dest_dir):
    for _, row in subset_df.iterrows():
        src = os.path.join(image_dir, row['filename'])
        dst = os.path.join(dest_dir, row['filename'])
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {row['filename']}: {e}")

print("Copying benchmark_dev images...")
copy_images(benchmark_dev_df, benchmark_dev_dir)
print("Copied benchmark_dev images.")

print("Copying benchmark_final images")
copy_images(benchmark_final_df, benchmark_final_dir)
print("Copied benchmark_final images.")

#Update metadata
df.loc[benchmark_dev_df.index, 'used_in_benchmark_dev'] = "yes"
df.loc[benchmark_final_df.index, 'used_in_benchmark_final'] = "yes"

#Save updated metadata
df.to_csv(metadata_path, index=False)
print("Metadata updated and saved.")

print("\nBenchmark split completed successfully!")
print(f"benchmark_dev: {len(benchmark_dev_df)} images")
print(f"benchmark_final: {len(benchmark_final_df)} images")
