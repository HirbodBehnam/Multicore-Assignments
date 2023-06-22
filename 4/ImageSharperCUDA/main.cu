#include <iostream>
#include <cassert>
#include "bmp.cuh"

#define BYTES_PER_PIXEL 3
#define WINDOW_SIZE 3
#define WINDOW_MIDDLE (WINDOW_SIZE / 2)

__constant__ uint32_t window[WINDOW_SIZE][WINDOW_SIZE];
__constant__ uint32_t window_sum;

struct pixel {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

__device__ void extract_pixel(const uint8_t *pixels_input, struct pixel *output_pixels, size_t i) {
    output_pixels->red = pixels_input[i];
    output_pixels->green = pixels_input[i + 1];
    output_pixels->blue = pixels_input[i + 2];
}

__device__ void
extract_window(const uint8_t *pixels_input, struct pixel output_pixels[WINDOW_SIZE][WINDOW_SIZE],
               int32_t row, int32_t column, uint32_t width, uint32_t height) {
    extract_pixel(pixels_input, &output_pixels[WINDOW_MIDDLE][WINDOW_MIDDLE], (row * width + column) * BYTES_PER_PIXEL);
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
                              (current_row * width + current_column) * BYTES_PER_PIXEL);
        }
}

__global__ void
weighting_average_thread(const uint8_t *input_pixels_cuda, uint8_t *output_pixels_cuda) {
    auto input_width = static_cast<uint32_t>(blockDim.x), input_height = static_cast<uint32_t>(gridDim.x);
    auto row = static_cast<int32_t>(blockIdx.x), column = static_cast<int32_t>(threadIdx.x);
    struct pixel output_pixels[WINDOW_SIZE][WINDOW_SIZE];
    extract_window(input_pixels_cuda, output_pixels, row, column, input_width, input_height);
    uint32_t red = 0, green = 0, blue = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = 0; j < WINDOW_SIZE; j++) {
            red += static_cast<uint32_t>(output_pixels[i][j].red) * window[i][j];
            green += static_cast<uint32_t>(output_pixels[i][j].green) * window[i][j];
            blue += static_cast<uint32_t>(output_pixels[i][j].blue) * window[i][j];
        }
    }
    red /= window_sum;
    green /= window_sum;
    blue /= window_sum;
    output_pixels_cuda[(row * input_width + column) * BYTES_PER_PIXEL] = red;
    output_pixels_cuda[(row * input_width + column) * BYTES_PER_PIXEL + 1] = green;
    output_pixels_cuda[(row * input_width + column) * BYTES_PER_PIXEL + 2] = blue;
}

int main() {
    // Read the input
    uint8_t *input_pixels;
    uint32_t input_width, input_height, input_bytesPerPixel;
    ReadImage("input.bmp", &input_pixels, &input_width, &input_height, &input_bytesPerPixel);
    assert(input_bytesPerPixel == BYTES_PER_PIXEL);
    std::cout << "Read input as " << input_width << "x" << input_height << " image" << std::endl;
    // Allocate CUDA resources
    uint8_t *input_pixels_cuda, *output_pixels_cuda;
    cudaMalloc(&input_pixels_cuda, input_width * input_height * BYTES_PER_PIXEL);
    cudaMalloc(&output_pixels_cuda, input_width * input_height * BYTES_PER_PIXEL);
    cudaMemcpy(input_pixels_cuda, input_pixels, input_width * input_height * BYTES_PER_PIXEL, cudaMemcpyHostToDevice);
    {
        uint32_t host_window[WINDOW_SIZE][WINDOW_SIZE] = {{1, 2, 1},
                                                          {2, 4, 2},
                                                          {1, 2, 1}};
        uint32_t host_window_sum = 16;
        cudaMemcpyToSymbol(window, host_window, sizeof(host_window), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(window_sum, &host_window_sum, sizeof(host_window_sum), 0, cudaMemcpyHostToDevice);
    }
    // Initialize the sharpening effect.
    // I know the dimensions and locality are bad but let's just do it, huh?
    weighting_average_thread<<<input_height, input_width>>>(input_pixels_cuda, output_pixels_cuda);
    auto thread_err = cudaDeviceSynchronize();
    if (thread_err != cudaSuccess) {
        std::cout << "Cannot execute tasks: " << cudaGetErrorString(thread_err) << std::endl;
        exit(1);
    }
    // Copy back to input (we don't need that buffer anymore)
    cudaMemcpy(input_pixels, output_pixels_cuda, input_width * input_height * BYTES_PER_PIXEL, cudaMemcpyDeviceToHost);
    cudaFree(input_pixels_cuda);
    cudaFree(output_pixels_cuda);
    // Save
    WriteImage("sharpen.bmp", input_pixels, input_width, input_height, input_bytesPerPixel);
    free(input_pixels);
    return 0;
}
