# 🧠 Image Sharpening Using Knowledge Distillation

This project aims to enhance image clarity in real-time applications (like video conferencing) by training a lightweight student model (NAFNet-Tiny) using knowledge distillation from a powerful teacher model (Restormer). It was developed as part of the **Intel Unnati Industrial Training 2025**.

---

## 📌 Motivation

- Traditional sharpening techniques (e.g., unsharp masking) often add artifacts and are not real-time friendly.
- Transformer-based models like **Restormer** are highly effective but computationally expensive.
- Goal: Use **Knowledge Distillation (KD)** to train a **compact model** that mimics Restormer's quality but runs efficiently on CPUs.

---

## 📁 Dataset

We curated a custom dataset of ~17,000 high-resolution images from:

- **DIV2K** – 800 images
- **Flickr2K** – 2,650 images
- **COCO** – 10,000+ images
- **Synthetic Text Images** – 2,000 images
- **Gaming Screenshots** – 1,500 images

Dataset available on request

### 💡 Preprocessing:

- Applied controlled blur: Bicubic downscaling + Gaussian blur
- Added JPEG compression and noise (subset-dependent)
- Extracted non-overlapping `256×256` patches
- Stored in `.png` format for consistency

---

## 📊 Subset Training Strategy

We followed a **multi-phase training approach** with 8 subsets simulating various degradations:

| Subset | Blur Type | JPEG | Noise | Purpose |
|--------|-----------|------|-------|---------|
| 1      | Heavy     | ✅   | ❌    | Initial training |
| 3      | Mixed     | ❌   | ❌    | Blur diversity |
| 5      | Mixed     | ✅   | ✅    | Realistic corruption |
| 7      | Complex   | ✅   | ✅    | High variance |

Each subset allowed the student model to gradually generalize to real-world distortions.

---

## 🏗️ Model Architecture

### 🔹 Teacher: [Restormer](https://github.com/swz30/Restormer)
- Transformer-based image restoration model
- Used official pretrained weights:
  - **Single Image Defocus Deblurring**
  - **Gaussian Color Denoising (Blind)**

### 🔹 Student: [NAFNet-Tiny](https://github.com/megvii-research/NAFNet)
- CNN-based lightweight architecture (~0.7M parameters)
- Trained using a distillation loss:
  
  ```python
  loss = α * loss_ground_truth + (1 - α) * loss_teacher
With α values like 0.7, 0.5 for different phases.

🧪 Training & Evaluation
Trained for 110 epochs

Intermediate validation used benchmark patch set

Final evaluation used 85 full-resolution blurred images (≥1920×1080)

✅ SSIM Scores (Full Images):
Blur Level	SSIM
Heavy	88
Medium	93
Low	95

✅ Survey Results:
Human users rated sharpened outputs (avg. score ~4.3/5)

Student model closely matched teacher outputs visually

🚀 Deployment
Student model exported to ONNX

Integrated with OBS Virtual Camera

Scripted real-time inference pipeline using OpenCV
to run depolyment
run: 
python deploy_sharpening.py

Runs in real time on CPU (~15–20 FPS) and can enhance webcam feed in Zoom, Meet, etc.

### Student model folder structure looks like
├── nafnet_tiny.py          # Student model definition
├── train_student.py        # Student training script
├── test_student.py         # Evaluation script
├── dataset.py              # Custom dataset loader
├── utils.py                # Utility functions (metrics, saves)
├── deployment.py           # ONNX conversion + webcam pipeline
├── deploy_sharpening.py    # Virtual webcam output
├── README.md               # Project overview

### 📚 References
Restormer GitHub

NAFNet GitHub

Intel® Unnati Program

ONNX Runtime

OBS Studio

### 🙌 Acknowledgements
Developed by:

Jahnavi Gondi

Bhuma Chaitanya Deepika

Ravula Cheruvu Sai Sruthi

Under the guidance of Prof. Meena K, GITAM School of Technology, Bengaluru.
