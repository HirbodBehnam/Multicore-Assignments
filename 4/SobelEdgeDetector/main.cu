#include <iostream>
#include "reader.cuh"
#include "brighter.cuh"
#include "saver.cuh"

int main(int argc, char **argv) {
    // Check arguments
    if (argc < 3) {
        std::cout << "Program usage:" << std::endl << argv[0] << " INPUT_NAME BRIGHTNESS_ALPHA" << std::endl;
        exit(1);
    }
    float alpha = strtof(argv[2], nullptr);
    std::cout << "Running with " << alpha << " as alpha" << std::endl;
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
    return 0;
}
