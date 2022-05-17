#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:4
#SBATCH --job-name=homography_training_mundus_hs_32
#SBATCH --output=homography_training_mundus_32.out
#SBATCH --error=homography_training_mundus_32.err

python main_rnn.py --cuda /mundus/abassioun830/data-driven/ShopFacade_depth
