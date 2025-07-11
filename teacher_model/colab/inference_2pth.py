import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

#  Config 
drive_root = '/content/drive/MyDrive/Image_Sharpening'
input_folder = "/content/blur_patches/blur_patches"
output_folder = os.path.join(drive_root, 'teacher_patches_dual')
os.makedirs(output_folder, exist_ok=True)

# Paths to pretrained models
defocus_path = os.path.join(drive_root, 'single_image_defocus_deblurring.pth')
denoise_path = os.path.join(drive_root, 'gaussian_color_denoising_blind.pth')

# Add Restormer repo path
restormer_root = '/content/Restormer'
sys.path.insert(0, restormer_root)

from basicsr.models.archs.restormer_arch import Restormer

#  Setup 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

#  Load Defocus Teacher 
model_def = Restormer().to(device)
ckpt_def = torch.load(defocus_path, map_location=device)
model_def.load_state_dict(ckpt_def['params'], strict=False)
model_def.eval()

#  Load Denoise Teacher 
model_den = Restormer().to(device)
ckpt_den = torch.load(denoise_path, map_location=device)
model_den.load_state_dict(ckpt_den['params'], strict=False)
model_den.eval()

#  Weighted Average Settings 
weight_def, weight_den = 0.7, 0.3

#  Process Blurred Patches 
png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

with torch.no_grad():
    for fname in tqdm(png_files, desc="Generating dual-teacher soft targets"):
        img_path = os.path.join(input_folder, fname)
        img = Image.open(img_path).convert("RGB")
        inp = to_tensor(img).unsqueeze(0).to(device)

        # Pad to multiple of 8
        h, w = inp.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        inp_padded = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

        # Inference from both teachers
        out_def = model_def(inp_padded)[:, :, :h, :w]
        out_den = model_den(inp_padded)[:, :, :h, :w]

        # Weighted fusion
        soft_target = torch.clamp(weight_def * out_def + weight_den * out_den, 0, 1)

        # Save PNG
        out_img = to_pil(soft_target.squeeze(0).cpu())
        out_img.save(os.path.join(output_folder, fname))

print("✅ All soft targets generated using both teachers (saved as PNG).")
