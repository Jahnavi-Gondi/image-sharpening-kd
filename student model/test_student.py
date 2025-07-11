import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from nafnet_tiny import get_nafnet_tiny
from utils import calculate_ssim, save_image_tensor
import numpy as np

# ---------- Config ----------
input_dir = r'C:\benchmark_2\benchmark2_blur\heavy'     # 🔁 CHANGE THIS
gt_dir = r'C:\benchmark_2\benchmark2_sharp\heavy'       # 🔁 CHANGE THIS
checkpoint_path = r'C:\NAFNet-main\student_epoch_110.pth'  # 🔁 Your best model path
output_dir = 'test_outputs'
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Metric Functions ----------
def calculate_mse(img1, img2):
    return torch.mean((img1 - img2) ** 2).item()

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0) - 10 * np.log10(mse)

# ---------- Dataset ----------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.files = sorted(os.listdir(input_dir))
        self.transform = ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        input_path = os.path.join(self.input_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)

        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        input_tensor = self.transform(input_img)
        gt_tensor = self.transform(gt_img)

        return input_tensor, gt_tensor, fname

# ---------- Load Model ----------
model = get_nafnet_tiny().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ---------- Load Data ----------
test_dataset = TestDataset(input_dir, gt_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------- Run Inference ----------
total_psnr = 0
total_ssim = 0
total_mse = 0

with torch.no_grad():
    for input_img, gt_img, fname in tqdm(test_loader, desc="Testing"):
        input_img = input_img.to(device)
        gt_img = gt_img.to(device)

        output = model(input_img)

        mse = calculate_mse(output, gt_img)
        psnr = calculate_psnr(output, gt_img)
        ssim_val = calculate_ssim(output[0], gt_img[0])

        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim_val

        save_image_tensor(output[0], os.path.join(output_dir, fname[0]))

# ---------- Results ----------
num_samples = len(test_loader)
avg_mse = total_mse / num_samples
avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples

print(f"\n✅ Test completed on {num_samples} images.")
print(f"📉 Average MSE:   {avg_mse:.6f}")
print(f"📈 Average PSNR:  {avg_psnr:.2f} dB")
print(f"📊 Average SSIM:  {avg_ssim:.4f}")
