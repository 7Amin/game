#!/bin/bash
#SBATCH --account=PAS0536
#SBATCH --job-name=CO_C_M0_d3
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=1 --gpu_cmode=exclusive
#SBATCH --mem=35gb


# cd /users/PAS0536/aminr8/construction/application/predict/
# module load python/3.9-2022.05
# module spider cuda
# module load cuda/11.5.2 miniconda3
# conda activate local
# conda update -n base -c defaults conda
#
# conda create -n local python=3.9
# conda install numpy
# source activate venv_jan
# PYTHONPATH=../.. python train_1.py -s="CO" -c="Denver" -m=0 -d=3 -b=128 -l=True
