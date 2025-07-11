# Patching Scripts

This folder contains scripts to extract 256×256 patches from full-resolution images. These patches are used to train and validate both teacher and student models.

---

## 📜 Script Descriptions

| Script | Description |
|--------|-------------|
| `patching_only.py` | Extracts fixed-size 256×256 patches without overlap. Suitable for grid-based extraction. |
| `patching_padding.py` | Similar to above but adds padding if image size is not divisible by patch size. Ensures no area is missed. |

---

## ⚙️ Patch Settings

- **Patch size**: 256×256 (RGB)
- **Overlap**: No overlap (stride = 256), or padded to fit
- **Sharp–blur alignment**: Naming is preserved to match corresponding blur and sharp pairs

---

## ✅ Notes

- Run after blurring step.
- Output can be structured as:  
  `sharp_patches/`, `blur_patches/`, `teacher_output_patches/` (if applicable)

