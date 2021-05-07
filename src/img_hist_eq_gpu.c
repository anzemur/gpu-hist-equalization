#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <CL/cl.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define HISTOGRAM_BINS      (256)
#define WORKGROUP_SIZE      (256)
#define MAX_SOURCE_SIZE	    (16384)
#define MAX_IMG_FORMAT_SIZE (5)
#define MAX_IMG_NAME_SIZE   (100)

/**
 * Image struct. 
 */
typedef struct _image
{
    int width;
    int height;
    int cpp;
    int size_px;
    int size_cpp;
    char *format;
    char *name;
    unsigned char *data;
} image;

/**
 * Loads image and extracts its data.
 * @param img_path Path to image file.
 */
image load_image(char *img_path)
{
    image img;
    img.format = (char*)malloc(MAX_IMG_FORMAT_SIZE);
    strcpy(img.format, strrchr(img_path, '.') + 1);
    if (!img.format)
    {
        fprintf(stderr, "Error: Invalid image format: %s!\n", img_path);
        exit(EXIT_FAILURE);
    }

    img.data = stbi_load(img_path, &img.width, &img.height, &img.cpp, 0);
    if (!img.data)
    {
        fprintf(stderr, "Error: Loading image '%s' failed!\n", img_path);
        exit(EXIT_FAILURE);
    }

    img.name = (char*)malloc(MAX_IMG_NAME_SIZE);
    strcpy(img.name, strrchr(img_path, '/') + 1);
    if (!img.name)
    {
        fprintf(stderr, "Error: Invalid image path: %s!\n", img_path);
        exit(EXIT_FAILURE);
    }

    img.size_px = img.width * img.height;
    img.size_cpp = img.size_px * img.cpp;

    return img;
}

/**
 * Saves image to image file based on given data.
 * @param filename File name.
 * @param format Image format.
 * @param width Image width
 * @param height Image height.
 * @param cpp Image channels per pixel.
 * @param img_data Image data.
 **/
void save_image(char *filename, char *format, int width, int height, int cpp, unsigned char *img_data)
{
    int status = 0;
    if (strcmp(format, "jpg") == 0 || strcmp(format, "jpeg") == 0)
    {
        status = stbi_write_jpg(filename, width, height, cpp, img_data, 100);
    } 
    else if (strcmp(format, "png") == 0)
    {
        status = stbi_write_png(filename, width, height, cpp, img_data, width * cpp);
    }
    else if (strcmp(format, "bmp") == 0)
    {
        status = stbi_write_bmp(filename, width, height, cpp, img_data);
    }
    else
    {
        fprintf(stderr, "Error: Invalid image format: %s!\n", format);
        exit(EXIT_FAILURE);
    }

    if (!status)
    {
        fprintf(stderr, "Error: There was a problem while saving image to a file.\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Loads kernel file and returns it as a string.
 * @param file_name Kernel file name.
 **/ 
char* load_kernel_file(char *file_name)
{
    FILE *fp;
    fp = fopen(file_name, "r");
    if (!fp)
    {
		fprintf(stderr, "Error: Opening kernel file: '%s' failed!\n", file_name);
        exit(EXIT_FAILURE);
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
    fclose(fp);

    return source_str;
}

int main(int argc, char **argv)
{
    char *img_path;
    if (argc > 1)
    {
        img_path = argv[1];
    }
    else
    {
        fprintf(stderr, "Error: Not enough arguments!\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Load image.
    image img = load_image(img_path);

    // Load kernel file. Filename is the same as program name.
    char *kernel_source = load_kernel_file(strncat(argv[0], ".cl", sizeof(argv[0]) + 3));

    // CL status var - used for saving status of OpenCL operations.
    cl_int cl_status;

    // Get platforms - OpenCL implementation (AMD, Intel, Nvidia).
    cl_uint num_platforms;
    cl_status = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    cl_status = clGetPlatformIDs(num_platforms, platforms, NULL);

    // Get platform devices - Actual GPUs/CPUs.
    cl_uint num_devices;
    cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    // Context - Brings everything together.
    cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &cl_status);

    // Command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &cl_status);

    // Create and build a program.
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&kernel_source, NULL, &cl_status);
    cl_status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

    // Build log.
    size_t build_log_len;
    char *build_log;
    cl_status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        build_log = (char *)malloc(sizeof(char)*(build_log_len+1));
        cl_status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return EXIT_FAILURE;
    }

    // Allocate memory on device and transfer data from host.
    cl_mem img_in_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     img.size_cpp*sizeof(unsigned char), img.data, &cl_status);

    cl_mem hist_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   3*HISTOGRAM_BINS*sizeof(unsigned int), NULL, &cl_status);

    cl_mem cdfs_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   3*HISTOGRAM_BINS*sizeof(unsigned int), NULL, &cl_status);

    cl_mem min_cdfs_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       3*sizeof(unsigned int), NULL, &cl_status);

    cl_mem img_out_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      img.size_cpp*sizeof(unsigned char), NULL, &cl_status);

    // Create kernels and set arguments.
    cl_kernel kernel_img_histogram = clCreateKernel(program, "img_histogram", &cl_status);
    cl_status  = clSetKernelArg(kernel_img_histogram, 0, sizeof(cl_mem), (void *)&img_in_d);
    cl_status |= clSetKernelArg(kernel_img_histogram, 1, sizeof(cl_mem), (void *)&hist_d);
    cl_status |= clSetKernelArg(kernel_img_histogram, 2, sizeof(cl_int), (void *)&img.size_px);
    cl_status |= clSetKernelArg(kernel_img_histogram, 3, sizeof(cl_int), (void *)&img.cpp);

    cl_kernel kernel_histogram_cdfs = clCreateKernel(program, "histogram_cdfs", &cl_status);
    cl_status  = clSetKernelArg(kernel_histogram_cdfs, 0, sizeof(cl_mem), (void *)&hist_d);
    cl_status |= clSetKernelArg(kernel_histogram_cdfs, 1, sizeof(cl_mem), (void *)&cdfs_d);
    cl_status |= clSetKernelArg(kernel_histogram_cdfs, 2, sizeof(cl_mem), (void *)&min_cdfs_d);
    cl_status |= clSetKernelArg(kernel_histogram_cdfs, 3, 3*HISTOGRAM_BINS*sizeof(unsigned int), NULL);

    cl_kernel kernel_correct_img = clCreateKernel(program, "correct_img", &cl_status);
    cl_status  = clSetKernelArg(kernel_correct_img, 0, sizeof(cl_mem), (void *)&img_in_d);
    cl_status |= clSetKernelArg(kernel_correct_img, 1, sizeof(cl_mem), (void *)&img_out_d);
    cl_status |= clSetKernelArg(kernel_correct_img, 2, sizeof(cl_mem), (void *)&cdfs_d);
    cl_status |= clSetKernelArg(kernel_correct_img, 3, sizeof(cl_mem), (void *)&min_cdfs_d);
    cl_status |= clSetKernelArg(kernel_correct_img, 4, sizeof(cl_int), (void *)&img.size_px);
    cl_status |= clSetKernelArg(kernel_correct_img, 5, sizeof(cl_int), (void *)&img.cpp);

    // Divide work & execute kernels.
    size_t local_item_size_hist = WORKGROUP_SIZE;
    size_t num_groups_hist = ((img.size_px - 1) / local_item_size_hist + 1);
	size_t global_item_size_hist = num_groups_hist * local_item_size_hist;
    cl_status = clEnqueueNDRangeKernel(command_queue, kernel_img_histogram, 1, NULL,						
								      &global_item_size_hist, &local_item_size_hist, 0, NULL, NULL);

    size_t local_item_size_cdfs = HISTOGRAM_BINS;
	size_t global_item_size_cdfs = 3 * HISTOGRAM_BINS;
    cl_status = clEnqueueNDRangeKernel(command_queue, kernel_histogram_cdfs, 1, NULL,						
								      &global_item_size_cdfs, &local_item_size_cdfs, 0, NULL, NULL);

    size_t local_item_size_correct = WORKGROUP_SIZE;
    size_t num_groups_hist_correct = ((img.size_px - 1) / local_item_size_correct + 1);
	size_t global_item_size_correct = num_groups_hist * local_item_size_correct;
    cl_status = clEnqueueNDRangeKernel(command_queue, kernel_correct_img, 1, NULL,						
								      &global_item_size_correct, &local_item_size_correct, 0, NULL, NULL);


    // Copy results back to host.
    unsigned char *img_out = (unsigned char *)malloc(img.size_cpp * sizeof(unsigned char));
    cl_status = clEnqueueReadBuffer(command_queue, img_out_d, CL_TRUE, 0,						
							        img.size_cpp*sizeof(unsigned char), img_out, 0, NULL, NULL);

    // Save image to a new file.
    save_image(img.name, img.format, img.width, img.height, img.cpp, img_out);

    // Release & free.
    cl_status = clFlush(command_queue);
    cl_status = clFinish(command_queue);
    cl_status = clReleaseKernel(kernel_img_histogram);
    cl_status = clReleaseKernel(kernel_histogram_cdfs);
    cl_status = clReleaseKernel(kernel_correct_img);
    cl_status = clReleaseProgram(program);
    cl_status = clReleaseMemObject(img_in_d);
    cl_status = clReleaseMemObject(hist_d);
    cl_status = clReleaseMemObject(cdfs_d);
    cl_status = clReleaseMemObject(min_cdfs_d);
    cl_status = clReleaseMemObject(img_out_d);
    cl_status = clReleaseCommandQueue(command_queue);
    cl_status = clReleaseContext(context);

  	free(devices);
    free(platforms);
    free(kernel_source);
    free(img_out);
    free(img.format);
    free(img.name);
    free(img.data);

    return EXIT_SUCCESS;
}