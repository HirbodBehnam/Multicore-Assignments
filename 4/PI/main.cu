#include <iostream>
#include <random>

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024
// We have a 2 here because each tread consumes two number (for x and y)
#define RANDOM_NUMBERS (BLOCK_SIZE * GRID_SIZE * 100 * 2)
#define ITERATIONS_PER_THREAD (RANDOM_NUMBERS / BLOCK_SIZE / GRID_SIZE)

__global__ void calculate_pi(const float *numbers, int *global_inside_counter) {
    int local_inside_counter = 0;
    size_t start_index = (threadIdx.x + blockIdx.x * BLOCK_SIZE) * ITERATIONS_PER_THREAD;;
    for (size_t i = 0; i < ITERATIONS_PER_THREAD; i += 2) {
        float x = numbers[start_index + i];
        float y = numbers[start_index + i + 1];
        if (x * x + y * y < 1) {
            local_inside_counter++;
        }
    }
    atomicAdd(global_inside_counter, local_inside_counter);
}

int main() {
    // At first generate random numbers
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(-1, 1);
    auto *random_numbers = new float[RANDOM_NUMBERS];
    std::cout << "Generating random numbers..." << std::endl;
    for (size_t i = 0; i < RANDOM_NUMBERS; i++)
        random_numbers[i] = dist(rng);
    // Move to GPU
    float *random_numbers_gpu;
    cudaMalloc(&random_numbers_gpu, sizeof(float) * RANDOM_NUMBERS);
    cudaMemcpy(random_numbers_gpu, random_numbers, sizeof(float) * RANDOM_NUMBERS, cudaMemcpyHostToDevice);
    delete[] random_numbers;
    // Run the kernel and calculate
    int *inside_counter;
    cudaMallocManaged(&inside_counter, 4);
    *inside_counter = 0;
    std::cout << "Calculating PI..." << std::endl;
    calculate_pi<<<GRID_SIZE, BLOCK_SIZE>>>(random_numbers_gpu, inside_counter);
    auto thread_err = cudaDeviceSynchronize();
    if (thread_err != cudaSuccess) {
        std::cout << "Cannot execute tasks: " << cudaGetErrorString(thread_err) << std::endl;
        exit(1);
    }
    // Free up resources
    cudaFree(random_numbers_gpu);
    // Print results
    std::cout << "From " << (RANDOM_NUMBERS / 2) << " points " << *inside_counter << " of them was inside the circle"
              << std::endl;
    float pi = static_cast<float>(*inside_counter) / static_cast<float>(RANDOM_NUMBERS / 2) * 4;
    std::cout << "PI is " << pi << std::endl;
    return 0;
}

