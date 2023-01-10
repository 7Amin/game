#!/bin/bash
#SBATCH --account=PAS0536
#SBATCH --job-name=transformer
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1 --gpu_cmode=exclusive
#SBATCH --mem=100gb

module load python/3.9-2022.05
module spider cuda
module load cuda/11.5.2 miniconda3
source activate venv_1
pip3 intall -r requirements.txt
cd /users/PAS0536/aminr8/game/RL
PYTHONPATH=. python3 runner.py -m "transformer" -u 24 -g "Breakout-v4" -b 8
