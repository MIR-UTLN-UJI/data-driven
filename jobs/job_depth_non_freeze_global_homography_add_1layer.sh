#!/bin/bash

#SBATCH --output=output_depth_non_freeze_global_homography_add_1layer.log
#SBATCH --error=error_depth_non_freeze_global_homography_add_1layer.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
python main_depth_non_freeze_global_homography_add_1layer.py /mundus/helkholy834/data_driven/datasets/ShopFacade_depth --loss 'global_homography'
wandb login --relogin


