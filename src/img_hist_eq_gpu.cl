#define HISTOGRAM_BINS 256

/**
 * Calculates image RGB colors histogram.
 * @param img Input image.
 * @param hist Image RGB histogram. Each color has bin size of `HISTOGRAM_BINS`.
 * @param img_size Image size in pixels. Also represents the size of the problem.
 * @param img_cpp Image channels per pixel. 
 */
__kernel void img_histogram(__global uchar *img,
                            __global uint *hist,
                            int img_size,
                            int img_cpp)
{
  int gid = get_global_id(0);

  while (gid < img_size * img_cpp)
  {
    int curr_channel = gid % img_cpp;
    if (curr_channel < 3) {
      atomic_inc(&hist[curr_channel * HISTOGRAM_BINS + img[gid]]);
    } 
    gid += get_global_size(0);
  }
}

/**
 * Calculates cumulative distribution (cdfs) of RGB histogram using Blelloch Scan
 * algorithm and finds minimum cdfs.
 * @param hist Image RGB histogram.
 * @param cdfs_hist Cumulative distributions of RGB histogram.
 * @param min_cdfs Minimum cumulative distributions of each RGB channel.
 * @param local_mem Local memory - every thread in workgroup can access it.
 */
__kernel void histogram_cdfs(__global uint *hist,
                             __global uint *cdfs_hist,
                             __global uint *min_cdfs,
                             __local uint *local_mem)
{
  int global_size = get_global_size(0);
  int local_size = get_local_size(0);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int color_channel = gid % 3;

  // Load data to local memory.
  local_mem[lid] = hist[gid];
  
  // First reduction phase.
  uint offset = 1;
  for(int step = local_size >> 1; step > 0; step >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < step)
    {
      int i = offset * (2 * lid + 1) - 1;
      int j = offset * (2 * lid + 2) - 1;
      local_mem[j] += local_mem[i];
    }
    offset <<= 1;
  }

  // Handle the last value and set up min cdfs.
  if(lid == 0)
  {
    local_mem[local_size - 1] = 0;
    min_cdfs[color_channel] = local_mem[0];
  }

  // Second reduction phase.
  for(int step = 1; step < local_size; step <<= 1)
  {
    offset >>= 1;
  
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < step)
    {
      int i = offset * (2 * lid + 1) - 1;
      int j = offset * (2 * lid + 2) - 1;

      uint tmp = local_mem[i];
      local_mem[i] = local_mem[j];
      local_mem[j] += tmp;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Find minimum cdfs and save them.
  if (min_cdfs[color_channel] > local_mem[gid] && local_mem[gid] != 0)
    min_cdfs[color_channel] = local_mem[gid];

  // Save values to cdfs
  if(gid < global_size - 1)
    cdfs_hist[gid] = local_mem[lid + 1];

  // Handle the last values.
  if (lid == local_size - 1)
    cdfs_hist[gid] = cdfs_hist[gid - 1] + hist[gid];
}

/**
 * Corrects image based on given cumulative distributions of RGB histogram.
 * @param img Input/original image.
 * @param corrected_img Corrected/output image.
 * @param cdfs_hist Cumulative distributions of RGB histogram.
 * @param min_cdfs Cumulative distributions of RGB histogram.
 * @param img_size Image size in pixels.
 * @param img_cpp Image channels per pixel.
 */
__kernel void correct_img(__global uchar *img,
                          __global uchar *corrected_img,
                          __global uint *cdfs_hist,
                          __global uint *min_cdfs,
                          int img_size,
                          int img_cpp)
{
  int gid = get_global_id(0);

  while (gid < img_size * img_cpp)
  {
    // Apply histogram equalization on every image channel.
    int curr_channel = gid % img_cpp;
    if (curr_channel < 3)
    {
      int px = curr_channel * HISTOGRAM_BINS + img[gid];
      corrected_img[gid] = round(((float)(cdfs_hist[px] - min_cdfs[curr_channel]) / (img_size - min_cdfs[curr_channel])) * (HISTOGRAM_BINS - 1));
    }
    // Copy alpha channel if it exists.
    else if (curr_channel == 3)
    {
      corrected_img[gid] = img[gid];
    }
    gid += get_global_size(0);
  }
}