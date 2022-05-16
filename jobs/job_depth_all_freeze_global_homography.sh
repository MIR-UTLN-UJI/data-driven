#!/bin/bash

#SBATCH --output=output_depth_all_freeze_global_homography.log
#SBATCH --error=error_depth_all_freeze_global_homography.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
python main_depth_all_freeze_global_homography.py /mundus/helkholy834/data_driven/datasets/ShopFacade_depth --loss 'global_homography'
wandb login --relogin
