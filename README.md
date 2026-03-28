# MoRE
A deep learning model for predicting protein–peptide interactions

## Introduction
We present a protein–peptide interaction modeling framework built upon ESM-2, integrating multi-task LoRA fine-tuning and LoRA Fusion. The model first applies parameter-efficient LoRA adapters to multiple pretraining objectives—covering both protein-level and residue-level tasks—and then fuses these adapters via learnable fusion weights into a unified multi-task module. This design enables efficient cross-task knowledge sharing and transfer, offering a lightweight yet powerful solution for protein–peptide interaction prediction.

## Installation
MoRE is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/).

This project is developed using python 3.9.18, and mainly requires the following libraries:

```python
numpy==1.26.4
scikit-learn==1.6.1
torch==2.7.0
transformers==4.52.3
peft==0.15.2
```

We provide Conda and pip configurations for easy environment reproduction. To set up the Conda environment(which typically takes around 10 minutes), run the following commands sequentially:

```bash
conda env create -f more.yaml -n more
conda activate more
pip install -r requirements.txt
```

The code has been developed and tested on  CentOS Linux 7/8.

## Usage

This repository provides three main scripts for training, validation, and testing:

- `train.py`: Two-stage training script (pretraining + joint training)
- `val.py`: Validation script for evaluating trained models
- `test.py`: Testing script for evaluating on multiple test datasets

For detailed usage instructions, see [docs/usage.md](docs/usage.md).


## Hardware requirements
Due to the computational complexity of the model, running the full dataset requires a dedicated GPU to accelerate training/inference.

- Minimum requirements:
    * CPU: 8 cores or higher
    * RAM: 32+ GB
    * GPU: An NVIDIA GPU with at least 40 GB of VRAM

- Hardware used for testing: The results presented in the manuscript were generated with the following specifications:
    * Intel Xeon Platinum 8358P×2
    * NVIDIA A800


## Citation
