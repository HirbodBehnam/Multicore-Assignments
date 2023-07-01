#ifndef SOBELEDGEDETECTOR_SAVER_H
#define SOBELEDGEDETECTOR_SAVER_H

#endif //SOBELEDGEDETECTOR_SAVER_H

#include <vector>

/**
 * Saves an grayscale buffer to PNG file
 * @param output_path The path to save the file
 * @param image The image buffer. MUST BE IN GPU
 * @param width Width of image
 * @param height Height of image
 */
void save_grayscale_image_gpu(const char *output_path, const uint8_t *image, int width, int height);