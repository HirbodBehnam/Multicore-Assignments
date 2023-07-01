#include "brighter.h"

static float clamp_pixel(float x) {
    return std::max(0.0f, std::min(255.0f, x));
}

void brighter(std::vector<std::vector<uint8_t>> &image_buffer, float alpha, float beta) {
    for (auto &row: image_buffer)
        for (auto &pixel: row)
            pixel = static_cast<uint8_t>(clamp_pixel(static_cast<float>(pixel) * alpha + beta));
}