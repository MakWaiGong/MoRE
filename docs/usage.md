# Usage Guide

This document provides detailed instructions for using the three main scripts in the MoRE repository:

- `train.py`: Training script with two-stage training (pretraining + joint training)
- `val.py`: Validation script for evaluating trained models
- `test.py`: Testing script for evaluating on multiple test datasets

## Table of Contents

- [Training (`train.py`)](#training-trainpy)
- [Validation (`val.py`)](#validation-valpy)
- [Testing (`test.py`)](#testing-testpy)
- [Configuration File](#configuration-file)
- [Data Format](#data-format)
- [Model Checkpoints](#model-checkpoints)

---

## Training (`train.py`)

The training script implements a two-stage training process:
1. **Pretraining Stage**: Freezes fusion LoRA parameters and trains only the new LoRA module (initialized with PPB pretrained weights)
2. **Joint Training Stage**: Unfreezes all parameters and jointly optimizes the entire model

### Basic Usage

```bash
python train.py \
    --config PPI-site/src/config.json \
    --save_dir ./results/more \
    --epochs 4 \
    --pretrain_epochs 2 \
    --batch_size 2 \
    --lr 1e-4 \
    --pretrain_lr 1e-4 \
    --r 4
```

### Arguments

| Argument | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | `PPI-site/src/config.json` | Path to configuration file |
| `--epochs` | int | 5 | Total number of training epochs |
| `--pretrain_epochs` | int | 2 | Number of epochs for pretraining stage |
| `--batch_size` | int | 2 | Batch size for training |
| `--lr` | float | 1e-4 | Learning rate for joint training stage |
| `--pretrain_lr` | float | 1e-4 | Learning rate for pretraining stage |
| `--save_dir` | str | None | Directory to save model checkpoints (defaults to config value) |
| `--result_dir` | str | None | Directory to save training metrics (defaults to config value) |
| `--r` | int | 4 | Number of PCA principal components for LoRA fusion |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--train_csv` | str | None | Path to training CSV file (defaults to config value) |
| `--val_csv` | str | None | Path to validation CSV file (defaults to config value) |
| `--test_csv` | str | None | Path to test CSV file for monitoring (optional) |

### Output Files

The training script saves the following files in `--save_dir`:

- `best_pretrain_model.pth`: Best model from pretraining stage
- `best_model.pth`: Best model from joint training stage
- `fusion.pth`: Fusion module state (including MLP parameters, weight matrices, and PCA cache)
- `classifier.pth`: Classifier head state
- `input_loras/`: Directory containing input LoRA weights
  - `input_lora_0.pth`
  - `input_lora_1.pth`
  - `input_lora_2.pth`
  - `input_lora_3.pth`

Training metrics are saved to `--result_dir/train/training_metrics.csv`.

### Examples

**Basic training with default settings:**
```bash
python train.py \
    --config PPI-site/src/config.json \
    --save_dir ./results/more
```

**Training with custom paths and hyperparameters:**
```bash
python train.py \
    --config PPI-site/src/config.json \
    --save_dir ./results/my_experiment \
    --result_dir ./results/my_experiment \
    --train_csv PPI-site/data/ft_train.csv \
    --val_csv PPI-site/data/ft_val.csv \
    --epochs 6 \
    --pretrain_epochs 3 \
    --batch_size 4 \
    --lr 5e-5 \
    --pretrain_lr 1e-4 \
    --r 4 \
    --seed 42
```

**Training with test set monitoring:**
```bash
python train.py \
    --config PPI-site/src/config.json \
    --save_dir ./results/more \
    --train_csv PPI-site/data/ft_train.csv \
    --val_csv PPI-site/data/ft_val.csv \
    --test_csv PPI-site/data/bitenet_test.csv \
    --epochs 4 \
    --pretrain_epochs 2
```

---

## Validation (`val.py`)

The validation script loads a trained model and evaluates it on a validation dataset.

### Basic Usage

```bash
python val.py \
    --save_dir ./results/more \
    --val_csv PPI-site/data/ft_val.csv
```

### Arguments

| Argument | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | `PPI-site/src/config.json` | Path to configuration file |
| `--save_dir` | str | **required** | Directory containing saved model checkpoints |
| `--result_dir` | str | None | Directory to save validation results (defaults to `save_dir/val_results`) |
| `--batch_size` | int | 2 | Batch size for validation |
| `--val_csv` | str | None | Path to validation CSV file (defaults to config value) |
| `--r` | int | 4 | Number of PCA principal components (must match training) |
| `--seed` | int | 42 | Random seed for reproducibility |

### Output Files

- `val_results/val_metrics.csv`: Validation metrics (accuracy, AUC, PR-AUC, etc.)

### Examples

**Basic validation:**
```bash
python val.py \
    --save_dir ./results/more \
    --val_csv PPI-site/data/ft_val.csv
```

**Validation with custom result directory:**
```bash
python val.py \
    --save_dir ./results/more \
    --val_csv PPI-site/data/ft_val.csv \
    --result_dir ./results/validation_results \
    --batch_size 4 \
    --r 4
```

---

## Testing (`test.py`)

The testing script loads a trained model and evaluates it on one or more test datasets. It automatically generates ROC/PR curves and AUC variance estimates for each test set.
It will take around 10 minutes

### Basic Usage

```bash
# Test on multiple datasets
python test.py \
    --save_dir ./results/more \
    --test_csvs PPI-site/data/bitenet_test.csv \
                 PPI-site/data/interpep_test.csv \
                 PPI-site/data/pepbind_test.csv \
                 PPI-site/data/pepnn_test.csv
```

### Arguments

| Argument | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | `PPI-site/src/config.json` | Path to configuration file |
| `--save_dir` | str | **required** | Directory containing saved model checkpoints |
| `--result_dir` | str | None | Directory to save test results (defaults to `save_dir/test_results`) |
| `--batch_size` | int | 2 | Batch size for testing |
| `--test_csvs` | list | None | Path(s) to test CSV file(s). If not specified, uses default 4 test sets (bitenet, interpep, pepbind, pepnn) |
| `--r` | int | 4 | Number of PCA principal components (must match training) |
| `--seed` | int | 42 | Random seed for reproducibility |

### Output Files

For each test dataset, the script creates a subdirectory in `test_results/test/<dataset_name>/` containing:

- `test_metrics.csv`: Test metrics (accuracy, AUC, PR-AUC, etc.)
- `auc_variance.txt`: AUC variance estimates (DeLong and Hanley & McNeil methods)
- `roc_curve.png`: ROC curve plot
- `pr_curve.png`: Precision-Recall curve plot

### Examples

**Test on a single dataset:**
```bash
python test.py \
    --save_dir ./results/more \
    --test_csvs PPI-site/data/bitenet_test.csv
```

**Test on multiple datasets:**
```bash
python test.py \
    --save_dir ./results/more \
    --test_csvs PPI-site/data/bitenet_test.csv \
                 PPI-site/data/interpep_test.csv \
                 PPI-site/data/pepbind_test.csv \
                 PPI-site/data/pepnn_test.csv
```

**Test with custom result directory:**
```bash
python test.py \
    --save_dir ./results/more \
    --result_dir ./final_test_results \
    --test_csvs PPI-site/data/bitenet_test.csv \
                 PPI-site/data/interpep_test.csv \
    --batch_size 4
```

**Use default test sets (bitenet, interpep, pepbind, pepnn):**
```bash
python test.py \
    --save_dir ./results/more
```

---

## Configuration File

All scripts use a unified configuration file (`PPI-site/src/config.json`) that contains:

- **common**: Shared settings
  - `model_path`: Path to the base ESM-2 model
  - `lora_paths`: List of paths to pretrained LoRA adapters (must be 4 LoRAs)
  - `lora_keys`: List of LoRA layer keys to fuse
  - `lora`: LoRA configuration (r, lora_alpha, target_modules, etc.)
  - `seed`: Random seed
  - `device`: Device to use (cuda/cpu)

- **more**: MoRE-specific settings
  - `save_dir`: Directory to save model checkpoints
  - `result_dir`: Directory to save results
  - `epochs`: Total number of training epochs
  - `batch_size`: Batch size
  - `learning_rate`: Learning rate
  - `r`: Number of PCA principal components

You can override any configuration value using command-line arguments.

### Example Configuration

```json
{
  "common": {
    "model_path": "/path/to/esm2_t36_3B_UR50D",
    "lora_paths": [
      "/path/to/BMP/adapter_model.safetensors",
      "/path/to/PPB/adapter_model.safetensors",
      "/path/to/Dist2NBS/adapter_model.safetensors",
      "/path/to/SWBindCount/adapter_model.safetensors"
    ],
    "lora_keys": [...],
    "lora": {
      "r": 8,
      "lora_alpha": 16,
      "target_modules": ["key", "value"],
      "lora_dropout": 0.3,
      "bias": "none"
    }
  },
  "more": {
    "save_dir": "./results/more",
    "result_dir": "./results/more",
    "epochs": 4,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "r": 4
  }
}
```

---

## Data Format

The CSV files should contain the following columns:

- `Prot_seq` or `prot_seq`: Protein sequence (amino acid sequence)
- `Pep_seq` or `pep_seq`: Peptide sequence (amino acid sequence)
- `label`: Binary string where each character corresponds to an amino acid position
  - `1` = interaction site
  - `0` = non-interaction site
  - The length of the label string should match the length of the protein sequence

### Example CSV Format

```csv
Prot_seq,Pep_seq,label
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,AAAAA,0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

---

## Model Checkpoints

The training script saves model checkpoints in the following structure:

```
save_dir/
├── best_pretrain_model.pth      # Best model from pretraining stage
├── best_model.pth                # Best model from joint training stage
├── fusion.pth                    # Fusion module state
│   ├── fusion_state_dict         # Complete fusion module state
│   ├── adaptive_fusion_state_dict
│   ├── new_lora_state_dict
│   ├── V_r_cache                 # PCA projection matrices
│   ├── v_mean_cache              # Mean vectors for PCA
│   ├── Z_init_cache              # Initial PCA projections
│   ├── lora_keys                 # LoRA layer keys
│   ├── N, r, detected_r         # Fusion parameters
│   ├── hidden_dims               # Hidden layer dimensions
│   ├── epoch, val_acc, val_auc   # Training metadata
│   └── config                    # Training configuration
├── classifier.pth                # Classifier head state
│   ├── classifier_state_dict
│   ├── epoch
│   └── val_acc
└── input_loras/                  # Input LoRA weights
    ├── input_lora_0.pth
    ├── input_lora_1.pth
    ├── input_lora_2.pth
    └── input_lora_3.pth
```

Both `val.py` and `test.py` automatically load these checkpoints when provided with `--save_dir`. The scripts will:

1. Load input LoRA weights from `input_loras/`
2. Load fusion module state from `fusion.pth`
3. Restore PCA cache and other fusion attributes
4. Load classifier head from `classifier.pth`

**Important Notes:**

- The `--r` parameter must match the value used during training
- The configuration file should be the same as used during training
- Ensure all checkpoint files are present in the `--save_dir` directory

---

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running scripts from the MoRE root directory and that `PPI-site/src` is accessible.

2. **Checkpoint loading errors**: Ensure all checkpoint files are present and the `--r` parameter matches training.

3. **CUDA out of memory**: Reduce `--batch_size` or use gradient accumulation.

4. **LoRA key mismatches**: Verify that the LoRA paths in the config file are correct and all 4 LoRAs are present.

5. **Data format errors**: Ensure CSV files have the correct columns (`Prot_seq`/`prot_seq`, `Pep_seq`/`pep_seq`, `label`).

---

## Additional Resources

For more information about the MoRE model architecture and training methodology, please refer to the main paper and supplementary materials.

