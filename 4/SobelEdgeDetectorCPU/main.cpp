#include <iostream>
#include "brighter.h"
#include "reader.h"
#include "saver.h"
#include "sobel.h"

int main(int argc, char **argv) {
    // Check arguments
    if (argc < 2) {
        std::cout << "Program usage:" << std::endl << argv[0]
                  << " INPUT_NAME [BRIGHTNESS_ALPHA] [BRIGHTNESS_BETA] [THRESHOLD]"
                  << std::endl;
        exit(1);
    }
    float alpha = argc > 2 ? strtof(argv[2], nullptr) : 1.0f;
    float beta = argc > 3 ? strtof(argv[3], nullptr) : 0.0f;
    auto threshold = static_cast<int16_t>(argc > 4 ? strtol(argv[4], nullptr, 10) : 70);
    std::cout << "Running with " << alpha << " as alpha and " << beta << "as beta and " << threshold << " as threshold"
              << std::endl;
    // Read the image
    std::cout << "Reading input..." << std::endl;
    auto image = read_to_grayscale(argv[1]);
    save_grayscale_image("grayscale.png", image);
    // Brighten it
    std::cout << "Brightening input..." << std::endl;
    brighter(image, alpha, beta);
    save_grayscale_image("brighten.png", image);
    // Edge detect it
    sobel_edge_detection(image, threshold);
    return 0;
}
