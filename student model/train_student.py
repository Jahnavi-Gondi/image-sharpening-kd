import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nafnet_tiny import get_nafnet_tiny
from tqdm import tqdm
from dataset import DistillDataset
from utils import calculate_ssim, save_image_tensor
import os


# Configuration
total_epochs = 120
checkpoint_path = 'checkpoint_latest.pth'
start_epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = get_nafnet_tiny().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.MSELoss()

# 🗂️ Training Dataset
train_dataset = DistillDataset(
    input_dir=r'C:\blur_hard33',
    gt_dir=r'C:\sharp_hard33',
    teacher_dir=r'C:\teacher_hard33'
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 🗂️ Validation Dataset (no teacher_dir)
val_dataset = DistillDataset(
    input_dir=r'C:\benchmark1_blur_patch\heavy_patch',
    gt_dir=r'C:\benchmark1_sharp_patch\heavy_sharp'
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Folder for outputs
os.makedirs('epoch_outputs', exist_ok=True)

# 🔁 Resume from checkpoint
if os.path.exists(checkpoint_path):
    print(f"🔁 Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("🚀 Starting training from scratch")

#Validation function
def validate(model, val_loader):
    model.eval()
    total_val_loss = 0
    total_val_ssim = 0
    with torch.no_grad():
        for input_img, gt_img in val_loader:  # ← only 2 items now
            input_img = input_img.to(device)
            gt_img = gt_img.to(device)

            output = model(input_img)
            loss = criterion(output, gt_img)  # ← only GT loss

            total_val_loss += loss.item()

            for i in range(output.size(0)):
                total_val_ssim += calculate_ssim(output[i], gt_img[i])

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_ssim = total_val_ssim / len(val_loader.dataset)
    return avg_val_loss, avg_val_ssim


# 🧠 Training loop
for epoch in range(start_epoch, total_epochs):
    print(f"\n🔁 Starting Epoch {epoch + 1}")
    model.train()
    total_loss = 0
    total_ssim = 0

    for batch_idx, (input_img, gt_img, teacher_img) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        input_img = input_img.to(device)
        gt_img = gt_img.to(device)
        teacher_img = teacher_img.to(device)

        output = model(input_img)
        loss_gt = criterion(output, gt_img)
        loss_teacher = criterion(output, teacher_img)
        loss = 0.8 * loss_gt + 0.2 * loss_teacher

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        for i in range(output.size(0)):
            total_ssim += calculate_ssim(output[i], gt_img[i])

        if batch_idx == 0:
            save_image_tensor(output[0], f'epoch_outputs/epoch_{epoch + 1}_output.png')

    avg_train_loss = total_loss / len(train_loader)
    avg_train_ssim = total_ssim / len(train_loader.dataset)

    # 🔍 Validate
    val_loss, val_ssim = validate(model, val_loader)

    print(f"✅ Epoch {epoch + 1} completed | 🧪 Train Loss: {avg_train_loss:.4f}, SSIM: {avg_train_ssim:.4f} | 🔬 Val Loss: {val_loss:.4f}, SSIM: {val_ssim:.4f}", flush=True)

    # 💾 Save model and checkpoint
    torch.save(model.state_dict(), f'student_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)