#!/bin/bash
#SBATCH -A lingwang
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J baseline_group1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --chdir=/public/home/lingwang/MoRE/Pretrain/src
#SBATCH --output=/public/home/lingwang/MoRE/Pretrain/out/%j.BMP.out
#SBATCH --error=/public/home/lingwang/MoRE/Pretrain/out/%j.BMP.err

source ~/.bashrc
conda activate ft
nvidia-smi

mkdir -p /public/home/lingwang/MoRE/Pretrain/out

cd /public/home/lingwang/MoRE/Pretrain/src

echo "============================================================"
echo "BMP Pretraining Task"
echo "Start Time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python /public/home/lingwang/MoRE/Pretrain/src/BMP_pretrain.py \
    --config /public/home/lingwang/MoRE/Pretrain/src/pretrain_config.json

