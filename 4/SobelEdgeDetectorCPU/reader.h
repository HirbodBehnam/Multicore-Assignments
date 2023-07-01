#ifndef SOBELEDGEDETECTORCPU_READER_H
#define SOBELEDGEDETECTORCPU_READER_H

#endif //SOBELEDGEDETECTORCPU_READER_H

#include <cstdint>
#include <vector>

/**
 * Reads an image in grayscale format
 * @param filepath The filepath to read
 * @return The image read
 */
std::vector<std::vector<uint8_t>> read_to_grayscale(const char *filepath);