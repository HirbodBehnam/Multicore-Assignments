#ifndef SOBELEDGEDETECTORCPU_SOBEL_H
#define SOBELEDGEDETECTORCPU_SOBEL_H


#endif //SOBELEDGEDETECTORCPU_SOBEL_H

#include <cstdint>
#include <vector>

void sobel_edge_detection(const std::vector<std::vector<uint8_t>> &image, int16_t threshold);