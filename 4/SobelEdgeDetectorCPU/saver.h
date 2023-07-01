#ifndef SOBELEDGEDETECTORCPU_SAVER_H
#define SOBELEDGEDETECTORCPU_SAVER_H

#endif //SOBELEDGEDETECTORCPU_SAVER_H

#include <cstdint>
#include <vector>

/**
 * Save an image to a file
 * @param output_path The output path to save the file in.
 * @param image_buffer The image buffer to save
 */
void save_grayscale_image(const char *output_path, const std::vector<std::vector<uint8_t>> &image_buffer);