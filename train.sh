#!/bin/bash
  
#SBATCH --job-name=ae_sd
#SBATCH --output=ae_sd.out.%j
#SBATCH --error=ae_sd.out.%j
#SBATCH --time=24:00:00
#SBATCH --account=vulcan-abhinav
#SBATCH --partition=vulcan-dpart
#SBATCH --qos=vulcan-high
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem=32G


conda activate treering
python run_tree_ring_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14  --reference_model_pretrain laion2b_s12b_b42k
