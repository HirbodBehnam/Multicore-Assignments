#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "bmp.c"
#include <assert.h>
#include <immintrin.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef unsigned long long uint64;
typedef short int16;
typedef unsigned char byte;

bool is_green(uint64 pixel) {
    __m64 a = _mm_set_pi64x((long long) pixel);
    __m64 b = _mm_set_pi64x(_lrotl(pixel, 16));
    __m64 c = _mm_set_pi64x(_lrotl(pixel, 32));
    __m64 result = _m_paddw(a, b);
    result = _m_paddw(result, c);
    uint64 result_int = (uint64) result;
    uint64 green_dist = (result_int >> 0) & 65535;
    uint64 blue_dist = (result_int >> 16) & 65535;
    uint64 red_dist = (result_int >> 48) & 65535;
    return (green_dist < red_dist && green_dist < blue_dist);
}

uint64 extract_pixel(const byte *start_of_pixel) {
    uint64 result = 0;
    result |= ((uint64) start_of_pixel[0]) << 0;
    result |= ((uint64) start_of_pixel[1]) << 16;
    result |= ((uint64) start_of_pixel[2]) << 32;
    //result |= ((uint64) 255) << 48; No need I guess
    return result;
}

int main() {
    /* start reading the file and its information*/
    byte *pixels_top, *pixels_bg;
    int32 width_top, width_bg;
    int32 height_top, height_bg;
    int32 bytesPerPixel_top, bytesPerPixel_bg;
    ReadImage("dino.bmp", &pixels_top, &width_top, &height_top, &bytesPerPixel_top);
    ReadImage("parking.bmp", &pixels_bg, &width_bg, &height_bg, &bytesPerPixel_bg);

    /* images should have color and be of the same size */
    assert(bytesPerPixel_top == 3);
    assert(width_top == width_bg);
    assert(height_top == height_bg);
    assert(bytesPerPixel_top == bytesPerPixel_bg);

    /* we can now work with one size */
    int32 width = width_top, height = height_top, bytesPerPixel = bytesPerPixel_top;
    printf("%d bytes per pixel\n", bytesPerPixel);
    printf("%d total pixels\n", width_top * height_top);

    /* start replacing green screen */
    for (int i = 0; i < height * width; i++) {
        /**
         * Here is how I encode data: We have a uint64 which contains 8 bytes (4 words)
         * 0-1 bytes are Red, 2-3 is Green and 4-5 is Blue. 6-7 byte is zero.
         * With this we can use https://www.felixcloutier.com/x86/paddb:paddw:paddd:paddq
         * and https://www.felixcloutier.com/x86/rcl:rcr:rol:ror to add r + g and such.
         */
        uint64 pixel = extract_pixel(&pixels_top[i * bytesPerPixel]);
        if (is_green(pixel)) {
            pixels_top[i * bytesPerPixel] = pixels_bg[i * bytesPerPixel];
            pixels_top[i * bytesPerPixel + 1] = pixels_bg[i * bytesPerPixel + 1];
            pixels_top[i * bytesPerPixel + 2] = pixels_bg[i * bytesPerPixel + 2];
        }
    }

    /* write new image */
    WriteImage("replaced.bmp", pixels_top, width, height, bytesPerPixel);

    /* free everything */
    free(pixels_top);
    free(pixels_bg);
    return 0;
}
