# Blur Generation Scripts

This folder contains scripts used to generate artificially blurred versions of clean (sharp) images. These simulate various video degradation types (compression, Gaussian blur, etc.) for use in student model training.

---

## 📜 Script Descriptions

| Script | Description |
|--------|-------------|
| `blur_heavy_only.py` | Applies strong bicubic + Gaussian blur (Phase 1 heavy blur). |
| `blur_medium_only.py` | Applies medium-level bicubic + Gaussian blur (Phase 2). |
| `blur_jpeg_only.py` | Adds JPEG compression artifacts to simulate low-bitrate blur. |
| `blur_mix_only.py` | Applies a combination of downscaling and blur. |
| `blur_mix_eval.py` | Used for evaluation-phase blur simulation (balanced variant). |
| `blur_mix_jpeg_noise.py` | Combines blur, JPEG compression, and Gaussian noise — used for teacher denoising variant. |

---

## 🧪 Input / Output

- **Input**: Folder of sharp `.png` images.
- **Output**: Blurred images in the same resolution.
- Output names are preserved for patch alignment.

---

## ✅ Notes

- Choose different scripts depending on training phase and blur severity.
- Ensure sharp and blur image naming matches for patching.
- All scripts can be modified to use `argparse` or hardcoded paths.
