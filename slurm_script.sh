#! /bin/sh

#SBATCH --job-name=img-hist-eq
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=out/res.log
#SBATCH --error=out/error.log

srun --gpus=1 -n1 --reservation=fri --error=error.log --output=res.log img_hist_gpu img/8k.jpg