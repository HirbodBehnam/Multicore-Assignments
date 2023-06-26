#include <opencv2/opencv.hpp>
#include <iostream>

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024

/**
 * How much data should each thread consume?
 */
__constant__ size_t iterations_per_thread;

__global__ void invert_part(uchar *image, size_t size) {
    size_t start_index = (threadIdx.x + blockIdx.x * BLOCK_SIZE) * iterations_per_thread;
    for (size_t i = 0; i < iterations_per_thread; i++) {
        if (i + start_index >= size) // watch out for last thread
            break;
        // Invert the pixel
        image[i + start_index] = 255 - image[i + start_index];
    }
}

int main() {
    // Load the image
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cout << "Cannot load image." << std::endl;
        exit(1);
    }
    // Move data around
    const auto image_data_size = static_cast<size_t>(image.dataend - image.datastart);
    std::cout << "Read image with size of " << image_data_size << std::endl;
    {
        const size_t iterations_per_thread_host = image_data_size / GRID_SIZE / BLOCK_SIZE + 1;
        cudaMemcpyToSymbol(iterations_per_thread, &iterations_per_thread_host, sizeof(iterations_per_thread_host), 0,
                           cudaMemcpyHostToDevice);
    }
    uchar *cudaImageData;
    cudaMalloc(&cudaImageData, image_data_size * sizeof(uchar));
    cudaMemcpy(cudaImageData, image.data, image_data_size * sizeof(uchar), cudaMemcpyHostToDevice);
    // Create the kernel
    invert_part<<<GRID_SIZE, BLOCK_SIZE>>>(cudaImageData, image_data_size);
    auto thread_err = cudaDeviceSynchronize();
    if (thread_err != cudaSuccess) {
        std::cout << "Cannot execute tasks: " << cudaGetErrorString(thread_err) << std::endl;
        exit(1);
    }
    // Read back data from GPU
    cudaMemcpy(image.data, cudaImageData, image_data_size * sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaFree(cudaImageData);
    // Save
    cv::imwrite("output.jpg", image);
    return 0;
}
