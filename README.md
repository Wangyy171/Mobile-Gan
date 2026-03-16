# MobileGAN: A Lightweight Recognition-Aware Framework for Real-Time Underwater Image Enhancement

This repository provides the implementation of **MobileGAN**, a lightweight recognition-aware framework for real-time underwater image enhancement. The project includes the original MobileGAN baseline, the journal-version revised MobileGAN, and related evaluation scripts for quantitative testing, efficiency analysis, and ablation studies.

## 1. Project Overview

Underwater images often suffer from color distortion, low contrast, and blurred structural details due to wavelength-dependent light attenuation and scattering. To address these issues under deployment-constrained settings, this project focuses on a lightweight enhancement framework suitable for embedded underwater vision systems.

The repository currently supports:

- **Original MobileGAN**
- **Revised MobileGAN**
  - dual-reference feature consistency
  - edge-aware detail regularization
  - staged dynamic weighting
- **FUnIE-GAN baseline**
- Quantitative evaluation on **UIEB**
- Cross-dataset generalization evaluation on **EUVP**
- Efficiency evaluation including **Params**, **FPS**, and **Latency**

---

## 2. Environment

Recommended environment:

- Python 3.10
- PyTorch 2.0.0
- CUDA 11.8
- torchvision
- lpips
- Pillow
- NumPy < 2

A typical environment can be created in Conda, for example:

```bash
conda create -n uwgan python=3.10 -y
conda activate uwgan
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install lpips pillow "numpy<2" tqdm pyyaml scikit-image opencv-python
```

If NumPy import errors occur, reinstall a stable version such as:

```bash
pip install numpy==1.26.4
```

---

## 3. Repository Structure

A typical project structure is:

```text
UW_MobileGAN/
├─ train.py
├─ infer.py
├─ losses.py
├─ test_metrics.py
├─ test_fps.py
├─ config.yaml
├─ config_feat_only.yaml
├─ config_feat_edge.yaml
├─ run_train.py
├─ run_feat_only.py
├─ run_feat_edge.py
├─ run_full_revised.py
├─ models/
│  ├─ generator_mobile.py
│  ├─ generator_funie.py
│  └─ discriminator_patch.py
├─ data/
│  ├─ dataset.py
│  ├─ train/
│  │  ├─ LR/
│  │  └─ HR/
│  ├─ val/
│  │  ├─ LR/
│  │  └─ HR/
│  └─ EUVP/
│     └─ underwater_dark/
│        ├─ trainA/
│        └─ trainB/
├─ utils/
│  └─ metrics.py
├─ outputs/
└─ outputs_journal/
```

---

## 4. Datasets

### 4.1 UIEB

The main training and validation dataset is **UIEB**.

Expected directory format:

```text
data/
├─ train/
│  ├─ LR/
│  └─ HR/
└─ val/
   ├─ LR/
   └─ HR/
```

### 4.2 EUVP

Cross-dataset generalization evaluation is conducted on the **paired underwater_dark subset of EUVP**.

Expected directory format:

```text
data/
└─ EUVP/
   └─ underwater_dark/
      ├─ trainA/   # degraded / low-quality images
      └─ trainB/   # reference / high-quality images
```

In this project:

- `trainA` is used as the degraded image directory
- `trainB` is used as the reference image directory

---

## 5. Configurations

### 5.1 Main journal-version configuration

The main configuration file is:

```text
config.yaml
```

This configuration is used for the **full revised MobileGAN**, including:

- dual-reference feature consistency
- edge-aware loss
- staged dynamic weighting

### 5.2 Ablation configurations

Recommended ablation configs:

- `config_feat_only.yaml`
- `config_feat_edge.yaml`

You may also create:

- `config_original_mobilegan.yaml`

to reproduce the original MobileGAN baseline under the same unified pipeline.

---

## 6. Training

### 6.1 Full revised MobileGAN

You may run the main journal-version model by:

```bash
python train.py --cfg config.yaml --device cuda
```

Or directly run:

```bash
python run_full_revised.py
```

### 6.2 Ablation: feature consistency only

```bash
python run_feat_only.py
```

### 6.3 Ablation: feature consistency + edge-aware loss

```bash
python run_feat_edge.py
```

### 6.4 Original MobileGAN baseline

If you prepare `config_original_mobilegan.yaml`, you may run:

```bash
python train.py --cfg config_original_mobilegan.yaml --device cuda
```

---

## 7. Quantitative Evaluation

### 7.1 Evaluate on UIEB

```bash
python test_metrics.py --cfg config.yaml --ckpt PATH_TO_CHECKPOINT --dataset uieb --device cuda
```

### 7.2 Evaluate on EUVP

```bash
python test_metrics.py --cfg config.yaml --ckpt PATH_TO_CHECKPOINT --dataset euvp --device cuda
```

This script reports:

- PSNR
- SSIM
- NIQE (if available in the current environment)

---

## 8. Efficiency Evaluation

To evaluate parameter count, FPS, and latency:

```bash
python test_fps.py --cfg config.yaml --ckpt PATH_TO_CHECKPOINT --device cuda --name Revised_MobileGAN
```

You may similarly test:

- Original MobileGAN
- FUnIE-GAN
- Revised MobileGAN

under the same environment for a fair comparison.

---

## 9. Outputs

Training and evaluation results are automatically saved under the configured output directory.

Recommended output organization:

```text
outputs_journal/
├─ full_revised/
├─ original_mobilegan/
├─ feat_only/
├─ feat_edge/
└─ ...
```

Each experiment folder may contain:

```text
out_dir/
├─ checkpoints/
├─ samples/
├─ metrics/
└─ logs/
```

### 9.1 Checkpoints

Typical saved checkpoints include:

- `best_psnr_G.pth`
- `best_ssim_G.pth`
- `last_G.pth`

### 9.2 Training samples

Training-time visual samples are saved under:

```text
out_dir/samples/
```

For example:

- `ep001.png`
- `ep005.png`
- `ep030.png`
- `ep050.png`

### 9.3 Metrics

Validation metrics are saved under:

```text
out_dir/metrics/val_metrics.csv
```

Testing results are saved under:

```text
out_dir/metrics/test_metrics.csv
out_dir/metrics/fps_metrics.csv
```

---

## 10. Model Variants in This Project

### 10.1 FUnIE-GAN

Used as an **external baseline**.

### 10.2 Original MobileGAN

Used as the **internal baseline** corresponding to the original conference-version model.

### 10.3 Revised MobileGAN

Used as the **journal-version full model**, which includes:

- dual-reference feature consistency
- edge-aware loss
- dynamic weighting

---

## 11. Notes on Baseline Re-Implementation

Some compared baselines in this project were **re-implemented in Python** according to the architectural and training descriptions reported in the original papers, rather than directly using an official Python release. Therefore:

- slight deviations from originally reported results may exist
- parameter count and FPS may differ due to implementation and platform differences

All comparisons in this repository should be interpreted under the **same local evaluation environment and unified pipeline**.

---

## 12. Suggested Files to Release on GitHub

Recommended files to upload:

- `train.py`
- `test_metrics.py`
- `test_fps.py`
- `infer.py`
- `losses.py`
- `models/`
- `data/dataset.py`
- `utils/metrics.py`
- `config.yaml`
- `config_feat_only.yaml`
- `config_feat_edge.yaml`
- `README.md`

Optional:

- selected checkpoints
- example outputs
- CSV logs

Do **not** upload:

- `.idea/`
- `__pycache__/`
- temporary cache files
- unnecessarily large intermediate checkpoints
- unrelated local experiment folders

---

## 13. Citation

If you use this code or build upon this project, please cite the corresponding paper after publication.

---

## 14. Contact

For project-related issues, please contact the corresponding author or repository maintainer after the repository is publicly released.
