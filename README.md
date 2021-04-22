Compile OpenCL programme
```bash
module load CUDA
gcc img_hist_gpu.c -O2 -lm -lOpenCL --openmp -o img_hist_gpu
srun --gpus=1 -n1 --reservation=fri --error=out/error.log --output=out/res.log img_hist_gpu img/8k.jpg

gcc img_hist_eq.c -O2 -lm --openmp -o img_hist_eq
srun --reservation=fri --error=out/error-serial.log --output=out/res-serial.log img_hist_eq img/8k.jpg
```