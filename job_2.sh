#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:3
#SBATCH --job-name=homography_training
#SBATCH --output=homography_training.out
#SBATCH --error=homography_training.err

python main_rnn.py --cuda /mundus/abassioun830/data-driven/ShopFacade_depth
