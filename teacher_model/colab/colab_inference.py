# 4. Import everything & setup paths
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Add Restormer to Python path
restormer_root = '/content/Restormer'
sys.path.insert(0, restormer_root)

from basicsr.models.archs.restormer_arch import Restormer

# ========== CONFIG ==========

# Change these according to your Drive structure
drive_root = '/content/drive/MyDrive/Image_Sharpening'  # or any folder you choose

input_folder = "/content/blur_patches/blur_patches"
output_png_folder = os.path.join(drive_root, 'teacher_patches')
output_pt_folder = os.path.join(drive_root, 'teacher_pt')
pretrained_model_path = os.path.join(drive_root, 'single_image_defocus_deblurring.pth')

os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_pt_folder, exist_ok=True)

# 5. Preprocessing
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# 6. Load model and weights (on GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Restormer().to(device)

checkpoint = torch.load(pretrained_model_path, map_location=device)
model.load_state_dict(checkpoint['params'], strict=True)
model.eval()

# Inference loop with progress bar
from tqdm import tqdm # Import tqdm
import os # Ensure os is imported
print(f"files: {len(input_folder)}")
png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
print(f"Total .png files found: {len(png_files)}")

# Ensure output directories exist
os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_pt_folder, exist_ok=True)

with torch.no_grad():
    for filename in tqdm(png_files, desc="Processing patches"):
        input_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(input_path).convert("RGB")
            img_tensor = to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)

            # Pad if needed to multiple of 8
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            output_tensor = model(img_tensor)

            # Unpad and clamp
            output_tensor = output_tensor[:, :, :h, :w].clamp(0, 1).cpu()

            # Save PNG
            output_image = to_pil(output_tensor.squeeze(0))
            output_png_path = os.path.join(output_png_folder, filename)
            output_image.save(output_png_path)

            # Save tensor
            output_pt_path = os.path.join(output_pt_folder, filename.replace('.png', '.pt'))
            torch.save(output_tensor.squeeze(0), output_pt_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


print("🎯 All patches processed and saved.")