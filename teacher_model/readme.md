# 🧠 Teacher Model: Restormer Inference

This folder contains all inference scripts and notebooks used to generate outputs from the pre-trained **Restormer** model. These outputs serve as supervision signals for the student model in our knowledge distillation setup.

---

## 🔍 Overview

Restormer is used here as a high-performing teacher model to produce sharp image outputs from blurred inputs. Only specific variants of the model were used:

| Task | Used? | Notes |
|------|-------|-------|
| Single Image Defocus Deblurring | ✅ Yes | Used in both Colab and local inference |
| Gaussian Color Denoising (Blind) | ✅ Yes | Used only in Colab |
| Motion Deblurring, Deraining, etc. | ❌ No | Not used |

---

## 🗂 Folder Structure

```text
teacher_model/
├── colab/
│   ├── Restormer_inference_defocus_deblur.ipynb
│   ├── Restormer_inference_defocus_denoise_deblur.ipynb
│
├── local_machine/
│   ├── inference_defocus_only.py         # ✅ used
│   └── inference_defocus_plus_denoise.py # Not used
│   └── Restormer/                        # 🔁 must be cloned here
│
└── README.md

🔁 Colab Inference (colab/)
These notebooks are fully self-contained and include:

Google Drive mounting

Cloning the official Restormer repo

Installing dependencies

Downloading pre-trained weights

Running inference

📓 Notebooks:
restormer_defocus_only.ipynb – defocus deblurring only

restormer_defocus_plus_denoise.ipynb – defocus + denoising

📌 You do not need to clone the Restormer repo manually in Colab. Everything is handled inside the notebook.

💻 Local Inference (local_machine/)
In local setups (e.g. VS Code), we only use the defocus deblurring model.

✅ Setup Instructions:
1.Clone the Restormer repo inside local_machine/:

  cd teacher_model/local_machine
  git clone https://github.com/swz30/Restormer.git

2.Download the defocus model weight from the Restormer repo:

  Restormer_Single_Image_Defocus_Deblurring.pth

3.Run the script:

  python inference_defocus_only.py

📌 You must manually set the input and output folder paths inside the script.


📤 Output Format
Input: 256×256 blurred patches or full-resolution images

Output: Corresponding sharpened images in .png format and .pt format

File names are preserved for alignment with ground truth and blur pairs


⚠️ Notes
Restormer is only used for inference — no training or fine-tuning was done.

Ensure consistent image alignment and naming for downstream distillation.

Local CPU inference is slower; use Colab when working with large datasets.
