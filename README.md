<div align="center">
<h2><center> 🌋 LaVA-Man: LaVA-Man: Learning Visual Action Representations for Robot Manipulation</h2>

[Chaoran Zhu](https://noone65536.github.io/)<sup>1</sup>, [Hengyi Wang](https://hengyiwang.github.io/)<sup>2</sup>,   [Yik Lung Pang](https://yiklungpang.github.io/)<sup>1</sup>, [Changjae Oh](https://eecs.qmul.ac.uk/~coh/)<sup>1</sup>

 <sup>1</sup>Queen Mary University of London, <sup>2</sup>University College London


<a href='https://www.arxiv.org/abs/2508.19391'><img src='https://img.shields.io/badge/ArXiv-2412.14803-red'></a> 
<a href='https://qm-ipalab.github.io/LaVA-Man/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>


This repo is the official PyTorch implementation for CoRL 2025 paper [**LaVA-Man: Learning Visual Action Representations for Robot Manipulation**](arxiv.org/abs/2508.19391).


## TODO list
- [x] Upload the model
- [x] Upload the environment setup instructions
- [ ] Upload the checkpoint and the inference scripts
- [ ] Upload the dataset and the preprocessed data
- [ ] Upload the training scripts

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/QM-IPAlab/LaVA-Man.git
cd LaVA-Man

# 1. Create and activate a new conda environment (Python 3.8+ recommended)
conda create -n lava-man python=3.8.6
conda activate myenv

# 2. Install PyTorch (This repo is tested with v2.4.0 and CUDA 12.4)
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install h5py
conda install -c conda-forge h5py

# 4. Install other dependencies
pip install -r requirements.txt
```


## 💾 Checkpoints

## 📊 Dataset

## 🏋️ Training

## 🔍 Inference Example

## 🙏 Acknowledgements
This repo builds on:
- CLIPort from https://github.com/cliport/cliport
- MAE from: https://github.com/facebookresearch/mae
- GroundingDINO from: https://github.com/IDEA-Research/GroundingDINO

## 📚 Citation