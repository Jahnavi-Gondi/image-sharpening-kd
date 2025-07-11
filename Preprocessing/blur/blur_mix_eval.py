import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import io

#config
input_dir = "path to sharp images(use from benchmark_validation)"  
output_dir = "path to copy blur images to"

#blur config
variant_config = {
    'heavy':  {'count': 28, 'bicubic_range': (1.7, 2.2), 'gaussian_range': (1.2, 1.6)},
    'medium': {'count': 28, 'bicubic_range': (1.4, 1.6), 'gaussian_range': (0.8, 1.1)},
    'low':    {'count': 28, 'bicubic_range': (1.1, 1.3), 'gaussian_range': (0.4, 0.7)},
}

jpeg_quality_range = (30, 60)  

all_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
random.shuffle(all_images)

start = 0
for variant, cfg in variant_config.items():
    variant_dir = os.path.join(output_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)

    selected_imgs = all_images[start:start + cfg['count']]
    start += cfg['count']


    for fname in tqdm(selected_imgs):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {fname}")
            continue

        h, w = img.shape[:2]

        # Bicubic down → up
        scale = round(random.uniform(*cfg['bicubic_range']), 2)
        down = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_CUBIC)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

        #Gaussian blur
        sigma = round(random.uniform(*cfg['gaussian_range']), 2)
        blurred = cv2.GaussianBlur(up, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # JPEG artifact 
        img_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        quality = random.randint(*jpeg_quality_range)
        img_pil.save(buffer, format="JPEG", quality=quality)


        buffer.seek(0)
        jpeg_artifacted = Image.open(buffer).convert("RGB")

        out_path = os.path.join(variant_dir, os.path.splitext(fname)[0] + '.png')
        jpeg_artifacted.save(out_path, format="PNG")

print("\nValidation images with blur + JPEG artifacts done!")
