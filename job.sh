#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:2
#SBATCH --job-name=homography_training
#SBATCH --output=homography_training.out
#SBATCH --error=homography_training.err

python main_rnn.py /mundus/aelsayed824/data-driven/ShopFacade_depth
