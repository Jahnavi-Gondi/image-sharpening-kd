# 📚 Student NAFNet Model (Tiny Version)

This repository contains a lightweight (student) version of the original [NAFNet](https://github.com/megvii-research/NAFNet) model designed for image restoration tasks, particularly image sharpening. The student model is trained for efficiency and deployed using ONNX for real-time inference.

---

student-model/

|-- NAFNet                        # must be cloned           
├── nafnet_tiny.py         
├── train_student.py              # Script to train the student model
├── test_student.py         
├── utils.py                
├── dataset.py              
├── deploy_sharpening.py     
├── deployment.py            
├── student_epoch_110.pth    
├── student_model.onnx       
├── README.md                
└── requirements.txt         




# 1. Clone this repository
git clone https://github.com/megvii-research/NAFNet


# 2. set the environment
$env:PYTHONPATH="C:\NAFNet"

# 3. Install required packages
pip install -r requirements.txt

# 4. run python train_student.py
the model will start training and the epochs and the epochs outputs are saved, in the training the validation images also we have to keep and after completing the epoch the trainind SSIM and validation SSIM will display

    # our results
    for student_epoch_110.pth
       - for heavy blur = SSIM : 85
       - for medium blur = SSIM : 91
       - for low blur = SSIM : 94

     # our dataset
      dataset/
      ├── train/
      │   ├── input/blur/
      │   └── gt/sharp/
      |   |__teacher/

      ├── val/
      │   ├── input/blur/
      │   └── gt/sharp/


# 5. run python test_student.py
to test the trained model select the best epoch and and test for the images

       # our dataset
       dataset/
       ├── test/
       │   ├── input/blur/
       │   └── gt/sharp/

     # our results
     for student_epoch_110.pth
       for patch images
           - for heavy blur = SSIM : 87
           - for medium blur = SSIM : 92
           - for low blur = SSIM : 94
       for full images 
           - for heavy blur = SSIM : 88
           - for medium blur = SSIM : 93
           - for low blur = SSIM : 95


# 6. run python deployment.py
the whole model is converted in to ONNX model after running this and it will be saved as student_model.onnx

# 7. run python deploy_sharpening.py 
to run this must instal "OBS studio" in the pc and after installing set up the OBS studio app in ur pc and close it
then run python deploy_sharpening.py the OBS camera will open automatically and the inference time and FPS is calculate

  - as we are running it on CPU we got 
                FPS = 15 - 20
  - this model will cross above 30 if is is run on high GPU





















