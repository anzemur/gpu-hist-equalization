# Histogram equalization

![](https://github.com/anzemur/img-hist-equalization/blob/main/example_img.png)

Parallel image histogram equalization on GPU with OpenCL. The program consists of three kernel functions - the first one for building the image histogram, the second one for calculating the cumulative distribution of the histogram (Blelloch Scan), and the final one for the image correction - equalization.


## How to use
```bash
module load CUDA
gcc img_hist_gpu.c -O2 -lm -lOpenCL --openmp -o img_hist_gpu
srun --gpus=1 -n1 --reservation=fri --error=out/error.log --output=out/res.log img_hist_gpu img/8k.jpg

gcc img_hist_eq.c -O2 -lm --openmp -o img_hist_eq
srun --reservation=fri --error=out/error-serial.log --output=out/res-serial.log img_hist_eq img/8k.jpg
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
