#!/bin/bash

#SBATCH -J test_job

#SBATCH -t 48:00:00

#SBATCH --partition=dept_gpu

#SBATCH --gres=gpu:1

#SBATCH --mem=64G



conda init bash

conda activate PPI-Transformer

python3 esm_generator.py