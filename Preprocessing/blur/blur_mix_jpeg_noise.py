import cv2
import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import io

#config
input_dir = "path to sharp images"
output_dir = "path to copy blur images to"
os.makedirs(output_dir, exist_ok=True)

# JPEG artifact 
jpeg_quality_range = (30, 60)
jpeg_artifact_prob = 0.7  # 70% prob

# Gaussian noise
noise_prob = 0.4           # 40% prob
noise_sigma_range = (2, 5)  

#Blur config
variant_config = {
    'heavy':  {'count': 235, 'bicubic_range': (1.7, 2.1), 'gaussian_range': (1.2, 1.6)},
    'medium': {'count': 215, 'bicubic_range': (1.4, 1.6), 'gaussian_range': (0.9, 1.1)},
    'low':    {'count': 100,  'bicubic_range': (1.0, 1.3), 'gaussian_range': (0.4, 0.8)},
}

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

        # Bicubic down → up
        scale = round(random.uniform(*cfg['bicubic_range']), 2)
        h, w = img.shape[:2]
        downscaled = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

        # Gaussian blur
        sigma = round(random.uniform(*cfg['gaussian_range']), 2)
        blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=sigma, sigmaY=sigma)

        #JPEG artifact
        apply_jpeg = random.random() < jpeg_artifact_prob
        if apply_jpeg:
            img_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            jpeg_quality = random.randint(*jpeg_quality_range)
            img_pil.save(buffer, format='JPEG', quality=jpeg_quality)
            buffer.seek(0)
            jpeg_img = Image.open(buffer).convert("RGB")
            final_img = cv2.cvtColor(np.array(jpeg_img), cv2.COLOR_RGB2BGR)
        else:
            final_img = blurred

        # OGaussian noise
        apply_noise = random.random() < noise_prob
        if apply_noise:
            noise_sigma = round(random.uniform(*noise_sigma_range), 2)
            noise = np.random.normal(0, noise_sigma, final_img.shape).astype(np.float32)
            noisy_img = np.clip(final_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            final_img = noisy_img

        out_path = os.path.join(variant_dir, os.path.splitext(fname)[0] + '.png')
        cv2.imwrite(out_path, final_img)

print("\ndone!")
