import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
restormer_root = os.path.join(current_dir, 'Restormer')  # relative to teacher_model

print("Current script dir:", current_dir)
print("Restormer root to add:", restormer_root)
print("sys.path before:", sys.path)

sys.path.insert(0, restormer_root)

print("sys.path after:", sys.path)

from basicsr.models.archs.restormer_arch import Restormer

# Configs
input_folder = r"C:\Users\janub\OneDrive\Desktop\Intel\Image Sharpening\Develop\preprocessing\data\2_preprocessed\subset1_500\test\blur_test_patches"
output_png_folder = r"C:\Users\janub\OneDrive\Desktop\Intel\Image Sharpening\Develop\preprocessing\data\2_preprocessed\subset1_500\test\teacher_test_png"
output_pt_folder = r"C:\Users\janub\OneDrive\Desktop\Intel\Image Sharpening\Develop\preprocessing\data\2_preprocessed\subset1_500\test\teacher_test_pt"
pretrained_model_path = r"C:\Users\janub\OneDrive\Desktop\Intel\Image Sharpening\Develop\teacher_model\Restormer\Defocus_Deblurring\pretrained_models\single_image_defocus_deblurring.pth"

os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_pt_folder, exist_ok=True)

# Image Preprocessing
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Load model
model = Restormer()
checkpoint = torch.load(pretrained_model_path, map_location='cpu')
model.load_state_dict(checkpoint['params'], strict=True)
model.eval()

# Inference
with torch.no_grad():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path).convert("RGB")
            img_tensor = to_tensor(img).unsqueeze(0) 

            # Pad if needed
            h, w = img_tensor.shape[2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            output_tensor = model(img_tensor)

            # Unpad
            output_tensor = output_tensor[:, :, :h, :w].clamp(0, 1)
            output_image = to_pil(output_tensor.squeeze(0))

            # Save PNG
            output_png_path = os.path.join(output_png_folder, filename)
            output_image.save(output_png_path)

            # Save Tensor
            output_pt_path = os.path.join(output_pt_folder, filename.replace(".png", ".pt"))
            torch.save(output_tensor.squeeze(0), output_pt_path)

            print(f"Processed: {filename}")

print("\nAll patches processed and saved.")
