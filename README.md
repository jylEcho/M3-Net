# M³Net
M³Net: A Macro→Meso→Micro Clinical-Guided Explainability Enhancing 3D Network for Pulmonary Nodule Classification

✨ Framework
<center>
  <img src="https://github.com/jylEcho/M3-Net/blob/main/images/framework.png" width="800" alt="">
</center>

🚀 Quick Start Guide
Environment Setup

To ensure reproducibility, we recommend using Conda to create a clean Python environment.

Operating System: Linux 5.4.0
Python Version: 3.8
CUDA Version: 11.8

1️⃣ **Create the Conda environment**
conda create -n liverseg_env python=3.8 -y
conda activate liverseg_env

2️⃣ **Install PyTorch (with CUDA 11.8 support)**
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

Verify installation:
python - << EOF
import torch
print(torch.__version__)
print(torch.cuda.is_available())
EOF

3️⃣ **Install required dependencies**
pip install \
  numpy==1.24.2 \
  scipy==1.10.1 \
  pandas==2.0.3 \
  tqdm==4.66.2 \
  scikit-image==0.21.0 \
  SimpleITK==2.2.1 \
  opencv-python==4.9.0.80 \
  einops \
  timm \
  safetensors

4️⃣ **(Optional) Install project-specific dependencies from GitHub**
git clone https://github.com/your-repo/liverseg.git
cd liverseg
pip install -e .

✅ **Notes**
Please do not install packages in the base Conda environment.
All experiments should be run inside the liverseg_env environment.
If CUDA is not detected, make sure the NVIDIA driver supports CUDA 11.8.

🏋️ **Training Pipeline**

1️⃣ Dataset preprocess
1、run the ClfDatasetPrepare_XXX.py to get the different cube sizes output
python ./ClfDatasetPrepare_XXX.py

2️⃣ Data_split
1、run the Data_split.py to split the Dataset into XX:XX:XX
python ./Data_split.py

3️⃣ Start Training!
python ./main.py

4️⃣ Start Testing!
python ./test.py

📊 Outputs
you can get the confusion matrix and other evalution ACC、Pre、Rec after testing.




## Acknowledgement

This project is based on [Malignancy-classification-in-LIDC-IDRI](https://github.com/jsyoonDL/Malignancy-classification-in-LIDC-IDRI) by [jsyoonDL](https://github.com/jsyoonDL).  
Some parts of the code have been modified and extended for my research/experiments. Thank You very Much!
