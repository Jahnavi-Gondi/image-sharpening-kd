import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import io

# CONFIG
input_dir = "path to sharp images"
output_dir = "path to copy blur images to"
os.makedirs(output_dir, exist_ok=True)

# JPEG artifacts
quality_range = (30, 60)

for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert("RGB")

    # Apply JPEG artifacts
    quality = np.random.randint(quality_range[0], quality_range[1] + 1)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)

    buffer.seek(0)
    jpeg_img = Image.open(buffer).convert("RGB")

    out_fname = fname.replace(".jpg", ".png").replace(".jpeg", ".png")
    out_path = os.path.join(output_dir, out_fname)
    jpeg_img.save(out_path, format="PNG")
