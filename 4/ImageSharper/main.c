#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "bmp.c"
#include <assert.h>
#include <immintrin.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0
#define WINDOW_SIZE 3
#define WINDOW_MIDDLE (WINDOW_SIZE / 2)
#define BYTES_PER_PIXEL 3

typedef unsigned int int32;
typedef unsigned long long uint64;
typedef short int16;
typedef unsigned char byte;

struct pixel {
    byte red;
    byte green;
    byte blue;
};

void extract_pixel(const byte *pixels_input, struct pixel *output_pixels, int32 i) {
    output_pixels->red = pixels_input[i];
    output_pixels->green = pixels_input[i + 1];
    output_pixels->blue = pixels_input[i + 2];
}

void
extract_window(const byte *pixels_input, struct pixel output_pixels[WINDOW_SIZE][WINDOW_SIZE], int row, int column,
               int32 width, int32 height) {
    extract_pixel(pixels_input, &output_pixels[WINDOW_MIDDLE][WINDOW_MIDDLE], (column * width + row) * BYTES_PER_PIXEL);
    for (int i = 0; i < WINDOW_SIZE; i++)
        for (int j = 0; j < WINDOW_SIZE; j++) {
            if (i == WINDOW_MIDDLE && j == WINDOW_MIDDLE) // don't do anything on middle
                continue;
            int current_row = row - (WINDOW_MIDDLE - i);
            int current_column = column - (WINDOW_MIDDLE - i);
            // If we are out of bounds, just use the center pixel
            if (current_row < 0 || current_row >= width || current_column < 0 || current_column >= height)
                output_pixels[i][j] = output_pixels[WINDOW_MIDDLE][WINDOW_MIDDLE];
            else
                extract_pixel(pixels_input, &output_pixels[i][j],
                              (current_column * width + current_row) * BYTES_PER_PIXEL);
        }
}

struct pixel weighting_average(const byte window_sum, const byte window[WINDOW_SIZE][WINDOW_SIZE],
                               const struct pixel image_window[WINDOW_SIZE][WINDOW_SIZE]) {
    uint64 red = 0, green = 0, blue = 0;
    for (int i = 0; i < WINDOW_SIZE; i++)
        for (int j = 0; j < WINDOW_SIZE; j++) {
            red += window[i][j] * image_window[i][j].red;
            green += window[i][j] * image_window[i][j].green;
            blue += window[i][j] * image_window[i][j].blue;
        }
    red /= window_sum;
    green /= window_sum;
    blue /= window_sum;
    struct pixel pixel = {.red = red, .green = green, .blue = blue};
    return pixel;
}

int main() {
    /* start reading the file and its information*/
    byte *pixels_input;
    int32 width_input;
    int32 height_input;
    int32 bytesPerPixel_input;
    ReadImage("input.bmp", &pixels_input, &width_input, &height_input, &bytesPerPixel_input);

    /* images should have color and be of the same size */
    assert(bytesPerPixel_input == BYTES_PER_PIXEL);
    printf("%d bytes per pixel\n", bytesPerPixel_input);
    printf("%d total pixels\n", width_input * height_input);
    byte *pixels_output = malloc(width_input * height_input * bytesPerPixel_input * sizeof(byte));

    // Define the window
    const byte window[WINDOW_SIZE][WINDOW_SIZE] = {{1, 2, 1},
                                                   {2, 4, 2},
                                                   {1, 2, 1}};
    const byte window_sum = 16;

    /* start replacing green screen */
    for (int i = 0; i < height_input; i++) {
        for (int j = 0; j < width_input; j++) {
            struct pixel image_window[WINDOW_SIZE][WINDOW_SIZE];
            extract_window(pixels_input, image_window, j, i, width_input, height_input);
            struct pixel result = weighting_average(window_sum, window, image_window);
            pixels_output[(i * width_input + j) * BYTES_PER_PIXEL] = result.red;
            pixels_output[(i * width_input + j) * BYTES_PER_PIXEL + 1] = result.green;
            pixels_output[(i * width_input + j) * BYTES_PER_PIXEL + 2] = result.blue;
        }
    }

    /* write new image */
    WriteImage("sharpen.bmp", pixels_output, width_input, height_input, bytesPerPixel_input);

    /* free everything */
    free(pixels_output);
    free(pixels_input);
    return 0;
}
