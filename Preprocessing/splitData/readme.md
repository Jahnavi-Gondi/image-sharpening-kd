# 📁 splitData

This folder contains scripts to filter and select subsets of images from a large dataset (~17,000 sharp images). These subsets are used throughout different phases of preprocessing and model training.

---

## 📜 Script Descriptions

| Script | Description |
|--------|-------------|
| `benchmark_split.py` | Picks diverse sharp images for final benchmark testing across categories (text, nature, people, etc.). |
| `subset_split.py` | Selects smaller subsets from the full dataset (e.g., `subset1`, `subset2`, `subset3` and so on) for different blur/training phases. |
| `train_test_split.py` | (Optional) Further splits selected subset into train/val/test sets. |
| `createCSV.py` | Generates CSV files listing filenames for each selected subset. Useful for loading and tracking datasets. |

---

## ✅ Notes

- This step happens **before any blurring or patching**.
- Subsets are manually or randomly sampled for use in different phases.
- Each subset is used independently for blur simulation and student/teacher training.
- Output can be folder copies or CSVs depending on your script.

---

## 🔁 Example Workflow

```text
1. Start with a large dataset of sharp images
2. Use `subset_split.py` to extract:
   - subset1 (for Phase 1)
   - subset2 (for Phase 2) and so on...
   - benchmark subset (for final testing)
3. Apply blur and patching to each subset separately
