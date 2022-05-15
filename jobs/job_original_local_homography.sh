#!/bin/bash

#SBATCH --job-name=original_model
#SBATCH --output=output_original_local_homography.log
#SBATCH --error=error_original_local_homography.log
#SBATCH --time=24:00:00

python main_original_local_homography.py /mundus/helkholy834/data_driven/datasets/ShopFacade_depth

