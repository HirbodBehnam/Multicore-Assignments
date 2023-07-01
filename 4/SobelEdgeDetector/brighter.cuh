#ifndef SOBELEDGEDETECTOR_BRIGHTER_CUH
#define SOBELEDGEDETECTOR_BRIGHTER_CUH

#endif //SOBELEDGEDETECTOR_BRIGHTER_CUH

/**
 * Makes an image brighter or darker
 * @param image The image to make brighter. Must be on GPU and in grayscale.
 * @param image_size The image size in bytes.
 * @param alpha How much brighter or darker? (1 biased)
 * @param beta How much brighter or darker?
 */
void brighter(uint8_t *image, size_t image_size, float alpha, float beta);