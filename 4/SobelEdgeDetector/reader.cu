#include <opencv2/opencv.hpp>
#include "reader.cuh"

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024
#define BYTES_PER_PIXEL 3

/**
 * Inverts an image and stores the result in @p grayscale_image
 * @param image The source image
 * @param grayscale_image The grayscale image goes here
 * @param image_size The number of bytes in @p image
 * @param iterations_per_thread Number of iterations each thread should do
 */
__global__ void
grayscale_image(const uchar *image, uint8_t *grayscale_image, const size_t image_size,
                const size_t iterations_per_thread) {
    const size_t grayscale_start_index = (threadIdx.x + blockIdx.x * BLOCK_SIZE) * iterations_per_thread;
    const size_t image_start_index = grayscale_start_index * BYTES_PER_PIXEL;
    for (size_t i = 0; i < iterations_per_thread; i++) {
        const size_t current_index = image_start_index + i * BYTES_PER_PIXEL;
        if (current_index >= image_size) // watch out for last thread
            break;
        // Grayscale the pixel. Used the opencv algorithm from
        // https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#void%20cvtColor%28InputArray%20src,%20OutputArray%20dst,%20int%20code,%20int%20dstCn%29
        // Color order is BGR
        float sum = 0;
        sum += static_cast<float>(image[current_index + 0]) * 0.114f; // B
        sum += static_cast<float>(image[current_index + 1]) * 0.587f; // G
        sum += static_cast<float>(image[current_index + 2]) * 0.299f; // R
        grayscale_image[grayscale_start_index + i] = static_cast<uint8_t>(sum);
    }
}

void read_to_grayscale(const char *filepath, uint8_t **data, int *width, int *height) {
    // Load the image
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cout << "Cannot load image." << std::endl;
        exit(1);
    }
    // Set the width and height
    *width = image.cols;
    *height = image.rows;
    // Convert to grayscale
    const auto image_data_size = image.cols * image.rows * BYTES_PER_PIXEL;
    // At first allocate stuff on GPU
    uchar *gpu_image;
    cudaMalloc(&gpu_image, image_data_size);
    cudaMalloc(data, image.cols * image.rows);
    cudaMemcpy(gpu_image, image.data, image_data_size, cudaMemcpyHostToDevice);
    // Now start the kernel
    const size_t number_of_iterations = (image_data_size / GRID_SIZE / BLOCK_SIZE / BYTES_PER_PIXEL) + 1;
    grayscale_image<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_image, *data, image_data_size, number_of_iterations);
    auto thread_err = cudaDeviceSynchronize();
    if (thread_err != cudaSuccess) {
        std::cout << "Cannot execute tasks: " << cudaGetErrorString(thread_err) << std::endl;
        exit(1);
    }
    // Free resources
    cudaFree(gpu_image);
}
