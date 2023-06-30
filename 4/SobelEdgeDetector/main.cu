#include <iostream>
#include "reader.cuh"
#include "brighter.cuh"
#include "saver.cuh"
#include "sobel.cuh"

int main(int argc, char **argv) {
    // Check arguments
    if (argc < 2) {
        std::cout << "Program usage:" << std::endl << argv[0] << " INPUT_NAME [BRIGHTNESS_ALPHA] [THRESHOLD]"
                  << std::endl;
        exit(1);
    }
    float alpha = argc > 2 ? strtof(argv[2], nullptr) : 1.0f;
    auto threshold = static_cast<int16_t>(argc > 3 ? strtol(argv[3], nullptr, 10) : 70);
    std::cout << "Running with " << alpha << " as alpha and " << threshold << " as threshold" << std::endl;
    // Read the image
    std::cout << "Reading input..." << std::endl;
    uint8_t *grayscale_image;
    int width, height;
    read_to_grayscale(argv[1], &grayscale_image, &width, &height);
    save_grayscale_image("grayscale.png", grayscale_image, width, height);
    // Brighten it
    std::cout << "Brightening input..." << std::endl;
    brighter(grayscale_image, width * height, alpha);
    save_grayscale_image("brighten.png", grayscale_image, width, height);
    // Edge detect it
    uint8_t *Gx, *Gy, *G;
    cudaMalloc(&Gx, width * height);
    cudaMalloc(&Gy, width * height);
    cudaMalloc(&G, width * height);
    sobel_edge_detection(grayscale_image, width, height, threshold, Gx, Gy, G);
    // Save images
    save_grayscale_image("Gx.png", Gx, width, height);
    save_grayscale_image("Gy.png", Gy, width, height);
    save_grayscale_image("G.png", G, width, height);
    // Cleanup
    cudaFree(G);
    cudaFree(Gy);
    cudaFree(Gy);
    cudaFree(grayscale_image);
    return 0;
}
