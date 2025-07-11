# student_training/utils.py

import torch
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ssim
import numpy as np

def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

def calculate_ssim(output, gt):
    output_np = tensor_to_numpy(output)
    gt_np = tensor_to_numpy(gt)
    return ssim(output_np, gt_np, channel_axis=2 ,data_range=1.0)

def save_image_tensor(img_tensor, path):
    img_tensor = torch.clamp(img_tensor, 0, 1)  # <- add this
    vutils.save_image(img_tensor, path)

