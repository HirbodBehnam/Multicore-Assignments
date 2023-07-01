#include <cmath>
#include "saver.h"
#include "sobel.h"

struct coordinates {
    int x, y;
};

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
    if (RESULT < 0)                                         \
        RESULT = 0;                                         \
} while(0)


static void
extract_window_cpu(const std::vector<std::vector<uint8_t>> &image, const struct coordinates cor, const int width,
                   const int height,
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
                result = image.at(current_cord.y).at(current_cord.x);
            }
            window[j + 1][i + 1] = result;
        }
}


void sobel_edge_detection(const std::vector<std::vector<uint8_t>> &image, int16_t threshold) {
    const float mx_cpu[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
    }, my_cpu[3][3] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
    };
    // Generate results
    int width = static_cast<int>(image[0].size());
    int height = static_cast<int>(image.size());
    std::vector<std::vector<uint8_t>> Gx(image.size(), std::vector<uint8_t>(image[0].size()));
    std::vector<std::vector<uint8_t>> Gy(image.size(), std::vector<uint8_t>(image[0].size()));
    std::vector<std::vector<uint8_t>> G(image.size(), std::vector<uint8_t>(image[0].size()));
    // Cast to image
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int16_t window[3][3];
            float Gx_elem, Gy_elem;
            extract_window_cpu(image, coordinates{.x = i, .y = j}, width, height, window);
            CONVOLUTION(window, mx_cpu, Gx_elem, threshold);
            CONVOLUTION(window, my_cpu, Gy_elem, threshold);
            Gx.at(j).at(i) = static_cast<uint8_t>(Gx_elem);
            Gy.at(j).at(i) = static_cast<uint8_t>(Gy_elem);
            G.at(j).at(i) = static_cast<uint8_t>(std::hypot(Gx_elem, Gy_elem));
        }
    }
    // Save them
    save_grayscale_image("Gx.png", Gx);
    save_grayscale_image("Gy.png", Gy);
    save_grayscale_image("G.png", G);
}