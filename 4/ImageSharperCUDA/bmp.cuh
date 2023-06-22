#ifndef IMAGESHARPERCUDA_BMP_CUH
#define IMAGESHARPERCUDA_BMP_CUH

#endif //IMAGESHARPERCUDA_BMP_CUH

void ReadImage(const char *fileName, uint8_t **pixels, uint32_t *width, uint32_t *height, uint32_t *bytesPerPixel);

void WriteImage(const char *fileName, uint8_t *pixels, uint32_t width, uint32_t height, uint32_t bytesPerPixel);