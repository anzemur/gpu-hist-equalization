#! /bin/sh

#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --reservation=fri
#SBATCH --output=logs/res.log
#SBATCH --error=error.log

srun --gpus=1 -n1 --reservation=fri --error=error.log --output=res.log img_hist_gpu img/8k.jpg