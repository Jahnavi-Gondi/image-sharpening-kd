import cv2
import os
import numpy as np
import random
from tqdm import tqdm

# config
input_dir = "path to sharp images"
output_dir = "path to copy blur images to"
os.makedirs(output_dir, exist_ok=True)

# blur
variant_config = {
    'heavy': {'count': 225, 'bicubic_range': (1.7, 2.0), 'gaussian_range': (1.2, 1.5)},
    'medium': {'count': 200, 'bicubic_range': (1.5, 1.5), 'gaussian_range': (0.8, 1.0)},
    'low': {'count': 75, 'bicubic_range': (1.2, 1.2), 'gaussian_range': (0.5, 0.6)},
}


all_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
random.shuffle(all_images)

start = 0
for variant, cfg in variant_config.items():
    variant_dir = os.path.join(output_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)

    selected_imgs = all_images[start:start + cfg['count']]
    start += cfg['count']

    print(f"\nApplying {variant} blur to {len(selected_imgs)} images...")

    for fname in tqdm(selected_imgs):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            print(f"Failed to read: {fname}")
            continue

        # Bicubic resize down → up
        scale = round(random.uniform(*cfg['bicubic_range']), 2)
        h, w = img.shape[:2]
        downscaled = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

        # Gaussian blur
        sigma = round(random.uniform(*cfg['gaussian_range']), 2)
        blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Save output
        out_path = os.path.join(variant_dir, fname)
        cv2.imwrite(out_path, blurred)

print("\nAll blur variants generated and saved.")
