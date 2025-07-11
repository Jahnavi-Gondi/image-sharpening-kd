import os
from PIL import Image
import numpy as np

def pad_image(img, patch_size=256):
    w, h = img.size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    pad_h = (patch_size - (h % patch_size)) % patch_size

    # Convert to numpy
    img_np = np.array(img)
    # Padding: (top, bottom), (left, right), (channels)
    padded_np = np.pad(img_np, 
    ((0, pad_h), (0, pad_w), (0, 0)), 
    mode='reflect')
    # Convert back to PIL
    padded_img = Image.fromarray(padded_np)
    return padded_img

def extract_patches(input_folder, output_folder, patch_size=256, stride=256):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path).convert("RGB")
            
            # Pad image first
            img = pad_image(img, patch_size)
            width, height = img.size

            patch_id = 0
            for top in range(0, height - patch_size + 1, stride):
                for left in range(0, width - patch_size + 1, stride):
                    patch = img.crop((left, top, left + patch_size, top + patch_size))
                    patch_filename = f"{os.path.splitext(filename)[0]}_patch{patch_id}.png"
                    patch.save(os.path.join(output_folder, patch_filename))
                    patch_id += 1

    print(f"Done extracting patches to {output_folder}")

base_path = "path to subset/folder of full images to patch"
sets = ['train', 'test']
patch_size = 256
stride = 256

for split in sets:
    sharp_input = os.path.join(base_path, split, f"sharp_{split}")
    sharp_output = os.path.join(base_path, split, f"sharp_{split}_patches")
    extract_patches(sharp_input, sharp_output, patch_size, stride)

    blur_input = os.path.join(base_path, split, f"blur_{split}")
    blur_output = os.path.join(base_path, split, f"blur_{split}_patches")
    extract_patches(blur_input, blur_output, patch_size, stride)







# import os
# from PIL import Image

# def extract_patches(input_folder, output_folder, patch_size=128, stride=128):
#     os.makedirs(output_folder, exist_ok=True)

#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".png"):
#             input_path = os.path.join(input_folder, filename)
#             img = Image.open(input_path).convert("RGB")
#             width, height = img.size

#             patch_id = 0
#             for top in range(0, height - patch_size + 1, stride):
#                 for left in range(0, width - patch_size + 1, stride):
#                     patch = img.crop((left, top, left + patch_size, top + patch_size))
#                     patch_filename = f"{os.path.splitext(filename)[0]}_patch{patch_id}.png"
#                     patch.save(os.path.join(output_folder, patch_filename))
#                     patch_id += 1

#     print(f"Done extracting patches to {output_folder}")

# # Example usage for both sharp and blurred folders:
# base_path = r"C:\Users\janub\OneDrive\Desktop\Intel\Image Sharpening\Develop\data\2_preprocessed\subset1_500"
# sets = ['train', 'test']
# patch_size = 128
# stride = 128

# for split in sets:
#     sharp_input = os.path.join(base_path, split, f"sharp_{split}")
#     sharp_output = os.path.join(base_path, split, f"sharp_{split}_patches")
#     extract_patches(sharp_input, sharp_output, patch_size, stride)

#     blur_input = os.path.join(base_path, split, "blur_1.7x_gaussian")
#     blur_output = os.path.join(base_path, split, "blur_1.7x_gaussian_patches")
#     extract_patches(blur_input, blur_output, patch_size, stride)
