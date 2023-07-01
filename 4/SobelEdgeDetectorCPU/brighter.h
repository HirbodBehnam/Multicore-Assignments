#ifndef SOBELEDGEDETECTORCPU_BRIGHTER_H
#define SOBELEDGEDETECTORCPU_BRIGHTER_H

#endif //SOBELEDGEDETECTORCPU_BRIGHTER_H

#include <cstdint>
#include <vector>

/**
 * Makes an image brighter
 * @param image_buffer The image to make it brighter
 * @param alpha
 * @param beta
 */
void brighter(std::vector<std::vector<uint8_t>> &image_buffer, float alpha, float beta);