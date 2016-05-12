#!/bin/bash
#SBATCH --job-name=hred-lstm
#SBATCH --output=log_files/hred-lstm.out
#SBATCH --error=log_files/hred-lstm.err
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -t 0-24:00:00
th train.lua -model_type hred -layer_type lstm -train_from models/lstm-subtle-fixed-we-2-layers_epoch3.00_97.88.t7