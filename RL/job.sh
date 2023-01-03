#!/bin/bash
#SBATCH --account=PAS0536
#SBATCH --job-name=class_sch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1 --gpu_cmode=exclusive
#SBATCH --mem=100gb
cd /users/PAS0536/aminr8/game/encoder_decoder
module load python/3.7-2019.10
module spider cuda
module load cuda/10.1.168 miniconda3
source activate venv
PYTHONPATH=.. python3 train.py -m "unet_vector"
