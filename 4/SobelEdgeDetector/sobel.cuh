#ifndef SOBELEDGEDETECTOR_SOBEL_CUH
#define SOBELEDGEDETECTOR_SOBEL_CUH

#endif //SOBELEDGEDETECTOR_SOBEL_CUH

/**
 * Do the sobel edge detection on an image
 * @param image The image to do the edge detection. Must be allocated on GPU.
 * @param width The width of image
 * @param height The height of image
 * @param threshold How much threshold should be applied for
 * @param Gx [OUT] The pointer to Gx variable after edge detection. Must be allocated on GPU.
 * @param Gy [OUT] The pointer to Gy variable after edge detection. Must be allocated on GPU.
 * @param G [OUT] The pointer to G variable after edge detection. Must be allocated on GPU.
 */
void sobel_edge_detection(const uint8_t *image, int width, int height, int16_t threshold, uint8_t *Gx, uint8_t *Gy,
                          uint8_t *G);

/**
 * Does the sobel edge detection but on CPU
 * @param gpu_image The image which must be on GPU
 * @param width Width of image
 * @param height Height of image
 * @param threshold The threshold of algorithm
 * @param show Show the edge detected image?
 */
void sobel_edge_detection_cpu(const uint8_t *gpu_image, int width, int height, int16_t threshold, bool show);