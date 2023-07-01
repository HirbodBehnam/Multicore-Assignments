#include <iostream>
#include "brighter.cuh"

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024

__device__ float clamp_pixel(float x) {
    return max(0.0f, min(255.0f, x));
}


/**
 * Makes an grayscale image brighter by multiplying its pixel by a value
 * @param grayscale_image The image
 * @param image_size The image size (in bytes)
 * @param iterations_per_thread The iterations which each thread should do
 * @param alpha How much to lighten or darken the image
 */
__global__ void
brighten_image(uint8_t *grayscale_image, const size_t image_size, const size_t iterations_per_thread,
               const float alpha, const float beta) {
    const size_t start_index = (threadIdx.x + blockIdx.x * BLOCK_SIZE) * iterations_per_thread;
    for (size_t i = 0; i < iterations_per_thread; i++) {
        const size_t current_index = start_index + i;
        if (current_index >= image_size)
            break;
        grayscale_image[current_index] = static_cast<uint8_t>(
                clamp_pixel(static_cast<float>(grayscale_image[current_index]) * alpha + beta));
    }
}

void brighter(uint8_t *image, size_t image_size, float alpha, float beta) {
    const size_t number_of_iterations = (image_size / GRID_SIZE / BLOCK_SIZE) + 1;
    brighten_image<<<GRID_SIZE, BLOCK_SIZE>>>(image, image_size, number_of_iterations, alpha, beta);
    cudaError_t thread_err = cudaDeviceSynchronize();
    if (thread_err != cudaSuccess) {
        std::cout << "Cannot execute tasks: " << cudaGetErrorString(thread_err) << std::endl;
        exit(1);
    }
}