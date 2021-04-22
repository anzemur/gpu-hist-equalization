#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define BINS 256

typedef struct _histogram
{
    unsigned int *R;
    unsigned int *G;
    unsigned int *B;
} histogram;

/**
 * Saves image to image file based on given data.
 **/
void save_image(char *filename, char *format, int width, int height, int cpp, unsigned char *img_data) {
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

    if (!status) {
        fprintf(stderr, "Error: There was a problem while saving image to a file.\n");
        exit(EXIT_FAILURE);
    }
}

unsigned int get_min_cdf(unsigned int *color_cdf)
{
    unsigned int min_cdf = color_cdf[0];
    for (int i = 1; i < BINS; i++)
    {
        if (color_cdf[i] < min_cdf && color_cdf[i] != 0)
        {
            min_cdf = color_cdf[i];
        }
    }
    return min_cdf;
}

unsigned char *histogram_eq(unsigned char *image, int width, int height, int cpp)
{
    //Initalize the histogram
    histogram H;
    H.B = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.G = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.R = (unsigned int *)calloc(BINS, sizeof(unsigned int));

    // Calculate image histogram.
    for (int i = 0; i < (height); i++)
    {
        for (int j = 0; j < (width); j++)
        {
            int px_idx = (i * width + j) * cpp;
            H.R[image[px_idx]]++;
            H.G[image[px_idx + 1]]++;
            H.B[image[px_idx + 2]]++;
        }
    }
   
    // Calculate cumulative distribution.
    for (int i = 1; i < BINS; i++)
    {
        H.R[i] += H.R[i - 1];
        H.G[i] += H.G[i - 1];
        H.B[i] += H.B[i - 1];
    }

    // Get minimum cdfs for each color channel.
    unsigned int min_cdf_R = get_min_cdf(H.R);
    unsigned int min_cdf_G = get_min_cdf(H.G);
    unsigned int min_cdf_B = get_min_cdf(H.B);

    // Create equalized image.
    unsigned char *image_eq = (unsigned char *)malloc(width * height * cpp * sizeof(unsigned char));
    for (int i = 0; i < (height); i++)
    {
        for (int j = 0; j < (width); j++)
        {
            int px_idx = (i * width + j) * cpp;
            int img_levels = BINS - 1;
            image_eq[px_idx] = round((((float)H.R[image[px_idx]] - min_cdf_R) / (width * height - min_cdf_R)) * img_levels);
            image_eq[px_idx + 1] = round((((float)H.G[image[px_idx + 1]] - min_cdf_G) / (width * height - min_cdf_G)) * img_levels);
            image_eq[px_idx + 2] = round((((float)H.B[image[px_idx + 2]] - min_cdf_B) / (width * height - min_cdf_B)) * img_levels);
            // Copy image alpha channel.
            if (cpp == 4)
                image_eq[px_idx + 3] = image[px_idx + 3];
        }
    }
    return image_eq;
}

int main(int argc, char **argv)
{
    char *image_file = argv[1];
    if (argc > 1)
    {
        image_file = argv[1];
    }
    else
    {
        fprintf(stderr, "Error: Not enough arguments!\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
        
    char *image_format = strrchr(image_file, '.') + 1;
    if (!image_format)
    {
        fprintf(stderr, "Error: Invalid image format ??: %s!\n", image_file);
        exit(EXIT_FAILURE);
    }

    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, 0);
    if (!image_in)
    {
        fprintf(stderr, "Error: Image loading failed: %s!\n", image_file);
        exit(EXIT_FAILURE);
    }
    double start = omp_get_wtime();
    unsigned char *image_out = histogram_eq(image_in, width, height, cpp);
    double end = omp_get_wtime();
    printf("Time: %f s\n", end-start);

    char *image_name = strrchr(image_file, '/') + 1;
    save_image(image_name, image_format, width, height, cpp, image_out);

    return EXIT_SUCCESS;
}