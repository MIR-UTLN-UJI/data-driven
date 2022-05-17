#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu
#SBATCH --job-name=homography_training_mundus
#SBATCH --output=homography_training_mundus.out
#SBATCH --error=homography_training_mundus.err

python main_rnn.py --cuda /mundus/abassioun830/data-driven/ShopFacade_depth
