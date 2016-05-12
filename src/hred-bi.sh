#!/bin/bash
#SBATCH --job-name=hred-bi
#SBATCH --output=log_files/hred-bi.out
#SBATCH --error=log_files/hred-bi.err
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -t 0-24:00:00
th train.lua -model_type hred -layer_type bi -train_from models/bi-subtle-fixed-we-2-layers_epoch3.00_85.98.t7 -load_red -gpuid 1