#ifndef SOBELEDGEDETECTOR_READER_CUH
#define SOBELEDGEDETECTOR_READER_CUH

#endif //SOBELEDGEDETECTOR_READER_CUH

/**
 * Reads an image and converts it to grayscale image
 * @param filepath The path of file
 * @param data [OUT] The pointer to data. The data will be allocated on GPU
 * @param width [OUT] Width of read image
 * @param height [OUT] Height of read image
 */
void read_to_grayscale(const char *filepath, uint8_t **data, int *width, int *height);