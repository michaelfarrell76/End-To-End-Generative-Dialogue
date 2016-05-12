#!/bin/bash
#SBATCH --job-name=hred-gru
#SBATCH --output=log_files/hred-gru.out
#SBATCH --error=log_files/hred-gru.err
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -t 0-24:00:00
th train.lua -model_type hred -layer_type lstm -train_from models/gru-subtle-fixed-we-2-layers_epoch3.00_93.24.t7