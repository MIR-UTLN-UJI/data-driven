#!/bin/bash

#SBATCH --job-name=original_model
#SBATCH --output=output_original_global_homography.log
#SBATCH --error=error_original_global_homography.log
#SBATCH --time=24:00:00

python main_global_homography.py /mundus/helkholy834/data_driven/datasets/ShopFacade_depth

