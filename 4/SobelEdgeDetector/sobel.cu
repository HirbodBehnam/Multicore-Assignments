#include <iostream>
#include "sobel.cuh"

// Mostly from https://github.com/fzehracetin/sobel-edge-detection-in-c

#define GRID_SIZE 1024
#define BLOCK_SIZE 1024

struct coordinates {
    int x, y;
};

__constant__ int16_t Mx[3][3], My[3][3];

/**
 * Extracts coordinates of an 1D index to 2D index on image
 * @param index The 1D index in image
 * @param width Width of image
 * @return The coordinates in 2D array
 */
__device__ struct coordinates extract_coordinates(const int index, const int width) {
    const struct coordinates result{
            .x = index % width,
            .y = index / width
    };
    return result;
}

__device__ size_t coordinates_to_index(const struct coordinates cor, const int width) {
    return cor.y * width + cor.x;
}

/**
 * Extracts a window for convolution.
 * @param image The image to extract the window from
 * @param width The width of image
 * @param height The height of image
 * @param window [OUT] The extracted window
 */
__device__ void extract_window(const uint8_t *image, const struct coordinates cor, const int width, const int height,
                               int16_t window[3][3]) {
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            const struct coordinates current_cord{
                    .x = cor.x + i,
                    .y = cor.y + j,
            };
            int16_t result = 0;
            if (current_cord.x >= 0 && current_cord.y >= 0 && current_cord.x < width && current_cord.y < height) {
                // in bounds of image
                result = image[coordinates_to_index(current_cord, width)];
            }
            window[i + 1][j + 1] = result;
        }
}

// My own way to use Mx and My without duplicating code and using pointers
#define CONVOLUTION(WINDOW, KERNEL, RESULT, THRESHOLD) do { \
    RESULT = 0;                                             \
    for (int x = 0; x < 3; x++){                            \
        for (int y = 0; y < 3; y++){                        \
            RESULT += WINDOW[x][y] * KERNEL[x][y];          \
        }                                                   \
    }                                                       \
    if (RESULT > THRESHOLD)                                 \
        RESULT = THRESHOLD;                                 \
} while(0);

__global__ void sobel_edge_detection_gpu(const uint8_t *image, const int width, const int height,
                                         const int16_t threshold, const size_t iterations_per_thread,
                                         uint8_t *Gx, uint8_t *Gy, uint8_t *G) {
    const int start_index = static_cast<int>((threadIdx.x + blockIdx.x * BLOCK_SIZE) * iterations_per_thread);
    const int image_size = width * height;
    for (int i = 0; i < iterations_per_thread; i++) {
        const int current_index = start_index + i;
        if (current_index >= image_size)
            break;
        // Extract the coordinates of current index
        const struct coordinates current_coordinates = extract_coordinates(current_index, width);
        int16_t window[3][3];
        int16_t Gx_elem, Gy_elem;
        extract_window(image, current_coordinates, width, height, window);
        CONVOLUTION(window, Mx, Gx_elem, threshold);
        CONVOLUTION(window, My, Gy_elem, threshold);
        Gx[current_index] = Gx_elem;
        Gy[current_index] = Gy_elem;
        G[current_index] = static_cast<uint8_t>(sqrt(static_cast<float>(Gx_elem * Gx_elem + Gy_elem + Gy_elem)));
    }
}

/**
 * Initializes @p Mx and @p My
 */
void init_mx_my() {
    const int16_t mx_cpu[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
    }, my_cpu[3][3] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
    };
    cudaMemcpyToSymbol(Mx, mx_cpu, sizeof(mx_cpu));
    cudaMemcpyToSymbol(My, my_cpu, sizeof(my_cpu));
}

void sobel_edge_detection(const uint8_t *image, int width, int height, int16_t threshold,
                          uint8_t *Gx, uint8_t *Gy, uint8_t *G) {
    // Create the Mx and My matrix
    init_mx_my();
    // Initialize the kernel
    const size_t number_of_iterations = (width * height / GRID_SIZE / BLOCK_SIZE) + 1;
    sobel_edge_detection_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(image, width, height, threshold, number_of_iterations,
                                                        Gx, Gy, G);
    cudaError_t thread_err = cudaDeviceSynchronize();
    if (thread_err != cudaSuccess) {
        std::cout << "Cannot execute tasks: " << cudaGetErrorString(thread_err) << std::endl;
        exit(1);
    }
}