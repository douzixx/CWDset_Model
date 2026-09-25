#  CWDset: A Remote Sensing Dataset for Semantic Segmentation of Small-Scale Coal Waste Dumps


## Key Features

* **Dataset:** 4-channel (R, G, B, NIR) high-resolution remote sensing imagery.
* **Custom Code:** Includes robust data loaders (`LoadTiffImageFromFile`,`LoadTiffAnnotations`) and augmentation pipelines (`RandomCropTiff`, `RandomFlipTiff`,`RandomResizeTiff`) **specifically designed for 4-channel TIF data**.
* **Benchmarks:** "Plug-and-play" configurations for 7 major segmentation models (BiSeNetV2, DeepLabV3+, Mask2Former, OCRNet, PSPNet, SegFormer, UNet).
* **Environment:** A 100% locked, reproducible, and stable Conda + PyTorch environment.

---

## 🔧 1. Installation
### Step 1: Download and Unzip Repository
1. **Download ZIP:** Click the green **< > Code** button, and then select **Download ZIP**.
2. **Rename and Prepare:** Unzip the downloaded file. 
### Step 2: Create the Conda Environment
```bash
# 1. Create a clean Python 3.10 environment
conda create -n mmlab_stable python=3.10 -y

# 2. Activate the new environment
conda activate mmlab_stable
```
### Step 3：Install PyTorch and mmcv-full
```bash
# 1. Install PyTorch 2.1.2 (LTS) + CUDA 12.1 Toolkit
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Install "mmcv-full" (v2.1.0)
pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```
### Step 4:Install All Remaining Dependencies
```bash
pip install -r requirements.txt
```
## 2. Data Preparation

1、The CWDset is officially hosted on Zenodo. You can access and download the dataset via the following link:
https://zenodo.org/records/22950251

2、Download the dataset.

3、Unzip the file and ensure your final project structure looks like this: 
```bash
pythonProject/
├── configs/
├── CWDset_code/
├── tools/
├── work_dirs/
├── data/
│   └── CWDset/
│       ├── images/
│       │   ├── train/
│       │   │   ├── img1.tif
│       │   │   └── ...
│       │   └── val/
│       │       └── ...
│       └── labels/
│           ├── train/
│           │   ├── img1.tif
│           │   └── ...
│           └── val/
│               └── ...
└── README.md
```
## 3.How to Reproduce
train  Models 
All 7 models are configured to train for 110,000 iterations and will only save the single best checkpoint based on val/mIoU.
All models were trained in the mmlab_stable environment on a single NVIDIA RTX 4080 SUPER for 110,000 iterations.

Example: Train BiSeNetV2: 
```bash
python tools/train.py configs/bisenetv2_fcn_b4_110k_CWDset-512x512.py
```
Testing (Evaluate mIoU) 
You can evaluate mIoU using either the weights you trained yourself, or our provided pre-trained weights.

Pre-trained Weights Location: We provide the final benchmark weights for all 7 models. During the review process, these can be downloaded from:
https://zenodo.org/records/22950251

Please download these weights. 

Example: Test BiSeNetV2 using our pre-trained weight: 
```bash
# Usage: python tools/test.py [CONFIG_FILE] [CHECKPOINT_FILE]
python tools/test.py configs/bisenetv2_fcn_b4_110k_CWDset-512x512.py pre-trained/bisenetv2.pth
```
Example: Test your self-trained BiSeNetV2 model: 
```bash
python tools/test.py configs/bisenetv2_fcn_b4_110k_CWDset-512x512.py work_dirs/bisenetv2_fcn_b4_110k_CWDset-512x512/best_val_mIoU_iter_XXXX.pth
```


