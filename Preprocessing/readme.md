# Preprocessing Pipeline

This folder contains the complete preprocessing pipeline used to prepare training and validation data for the image sharpening project. It includes scripts for selecting image subsets, applying blur, extracting patches, and preparing data splits.

---

## 🧭 Preprocessing Workflow

> The recommended order of execution is:

```text
1. Select subsets from the full image pool (~17,000 images)
   → Run scripts in `splitData/`

2. Generate blurred versions of sharp images
   → Run scripts in `blur/`

3. Extract 256×256 patches from sharp and blurred images
   → Run scripts in `patching/`

4. (Optional) Split patches into train/val or generate CSVs
   → Use `splitData/train_test_split.py` or `createCSV.py`
