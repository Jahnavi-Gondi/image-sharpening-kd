import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DistillDataset(Dataset):
    def __init__(self, input_dir, gt_dir, teacher_dir=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.teacher_dir = teacher_dir
        self.transform = transforms.ToTensor()

        input_files = set(os.listdir(input_dir))
        gt_files = set(os.listdir(gt_dir))
        common_files = input_files & gt_files

        if self.teacher_dir:
            teacher_files = set(os.listdir(teacher_dir))
            common_files = common_files & teacher_files

        self.valid_files = []
        for fname in sorted(common_files):
            try:
                _ = Image.open(os.path.join(self.input_dir, fname)).verify()
                _ = Image.open(os.path.join(self.gt_dir, fname)).verify()
                if self.teacher_dir:
                    _ = Image.open(os.path.join(self.teacher_dir, fname)).verify()
                self.valid_files.append(fname)
            except (UnidentifiedImageError, OSError) as e:
                print(f"❌ Skipping bad image: {fname} - {e}")

        if not self.valid_files:
            raise ValueError("No valid image files found.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        fname = self.valid_files[idx]
        input_img = self.transform(Image.open(os.path.join(self.input_dir, fname)).convert('RGB'))
        gt_img = self.transform(Image.open(os.path.join(self.gt_dir, fname)).convert('RGB'))

        if self.teacher_dir:
            teacher_img = self.transform(Image.open(os.path.join(self.teacher_dir, fname)).convert('RGB'))
            return input_img, gt_img, teacher_img
        else:
            return input_img, gt_img