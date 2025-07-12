#  Image Sharpening Using Knowledge Distillation

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

Note: There are few other datasets like GoPro,RealBlur (RealBlur-J / RealBlur-R), etc that you can use directly

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

###  Student Model: NAFNet-Tiny
NAFNet-Tiny is a lightweight CNN-based version of the original NAFNet, designed for real-time deployment and edge-device compatibility. It retains the core design of NAFNet but drastically reduces parameter count, making it much faster and memory-efficient.

| Feature               | Original NAFNet     | NAFNet-Tiny (Ours) |
|-----------------------|----------------------|---------------------|
| Parameters            | ~17.0M               | ~0.7M               |
| Feature Width         | 64                   | 16                  |
| Total NAFBlocks       | 72                   | 26                  |
| Middle Blocks         | 20                   | 2                   |
| FPS (CPU)             | Slow                 | 15–20 FPS           |
| ONNX Export           | ❌                   | ✅                  |
| Deployment Ready      | Not suitable         | ✅ Real-time         |


➡️ Inference Optimization: Despite being ~25× smaller than full NAFNet, the Tiny version preserves much of the teacher's visual fidelity.

🎯 Knowledge Distillation Strategy
The student model was trained using offline distillation from a pretrained Restormer teacher:
loss = α * loss_ground_truth + (1 - α) * loss_teacher
α was tuned across training phases:

Phase 1 (Heavy Blur): α = 0.7

Phase 2 (Mixed Blur): α = 0.5

Teacher outputs (.png) were generated using pretrained weights and served as soft supervision.

No normalization was applied — raw pixel values from .png were used directly to preserve fidelity.

🧪 Training & Evaluation
Training Duration: 110 epochs

Patch Size: 256×256

Patch-wise Validation: Done using 100 benchmark images during training

Final Evaluation: ~85 full-resolution (≥1920×1080) images with three blur levels:

Heavy: Bicubic 1.7–2.2 + Gaussian 1.2–1.5

Medium: Bicubic 1.5 + Gaussian 0.8

Low: Bicubic 1.2 + Gaussian 0.4

### 🔬 SSIM Comparison (Full Images)

| Blur Level | NAFNet-Tiny (Student) | Restormer (Teacher) |
|------------|------------------------|----------------------|
| Heavy      | 88                     | ~90                  |
| Medium     | 93                     | ~95                  |
| Low        | 95                     | ~96                  |
| **Average**| **≈92.0**              | ≈94.0–95.0           |


➡️ The student closely matched Restormer performance, especially on medium and low blur cases, with only ~4% parameter cost.

📊 Human Visual Evaluation (Survey)
Users were shown:

Blurred Input

Ground Truth

Teacher Output

Student Output

Asked to rate sharpness on a scale of 1–5

### 🧠 Visual Quality Survey (Average Ratings / 5)

| Output Type     | Average Rating |
|------------------|----------------|
| Blurred Image    | 3.0            |
| Teacher Output   | 3.7            |
| Student Output   | **4.3**        |
| Ground Truth     | 5.0            |


➡️ NAFNet-Tiny was consistently rated closer to ground truth, sometimes outperforming the teacher in perceived sharpness due to cleaner edges.

⚙️ Real-Time Ready
Converted to ONNX for efficient deployment

Runs at 15–20 FPS on CPU (Intel Core i5)

Compatible with:

OBS Studio virtual camera

Zoom / Google Meet / MS Teams

### 🚀 Deployment
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


## 📚 References

- [Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)](https://github.com/swz30/Restormer)
- [NAFNet: Nonlinear Activation Free Network for Image Restoration (ECCV 2022)](https://github.com/megvii-research/NAFNet)
- [Intel® Unnati Industrial Training Program](https://www.intel.com/content/www/us/en/education/unnati.html)
- [ONNX Runtime](https://onnxruntime.ai)
- [OBS Studio – Open Broadcaster Software](https://obsproject.com)
- [PyTorch Deep Learning Framework](https://pytorch.org)
- [OpenCV – Open Source Computer Vision Library](https://opencv.org)
- [COCO Dataset](https://cocodataset.org)


## 🙌 Acknowledgements
Developed by:

Jahnavi Gondi

Bhuma Chaitanya Deepika

Ravula Cheruvu Sai Sruthi

Under the guidance of Prof. Meena K, GITAM School of Technology, Bengaluru.
