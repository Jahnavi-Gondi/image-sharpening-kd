import os
from PIL import Image

#Configuration 
input_folder = "path to full image folder"
output_folder = "path to folder to copy patch images to"
patch_size = 256
stride = 256  

os.makedirs(output_folder, exist_ok=True)

def extract_patches(image_path, patch_size, stride, save_folder):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    patch_id = 0
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patch_filename = f"{base_name}_patch_{patch_id:04d}.png"
            patch.save(os.path.join(save_folder, patch_filename))
            patch_id += 1

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        extract_patches(image_path, patch_size, stride, output_folder)

print(f"Extracted 256x256 patches with stride 256 to: {output_folder}")
