#include <iostream>
#include "bmp.cuh"

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

//***Inputs*****
//fileName: The name of the file to open 
//***Outputs****
//pixels: A pointer to a byte array. This will contain the pixel data
//width: An int pointer to store the width of the image in pixels
//height: An int pointer to store the height of the image in pixels
//bytesPerPixel: An int pointer to store the number of bytes per pixel that are used in the image
void ReadImage(const char *fileName, uint8_t **pixels, uint32_t *width, uint32_t *height, uint32_t *bytesPerPixel) {
    //Open the file for reading in binary mode
    FILE *imageFile = fopen(fileName, "rb");
    //Read data offset
    uint32_t dataOffset;
    fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
    fread(&dataOffset, 4, 1, imageFile);
    //Read width
    fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
    fread(width, 4, 1, imageFile);
    //Read height
    fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
    fread(height, 4, 1, imageFile);
    //Read bits per pixel
    int16_t bitsPerPixel;
    fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
    fread(&bitsPerPixel, 2, 1, imageFile);
    //Allocate a pixel array
    *bytesPerPixel = ((uint32_t) bitsPerPixel) / 8;

    //Rows are stored bottom-up
    //Each row is padded to be a multiple of 4 bytes.
    //We calculate the padded row size in bytes
    int paddedRowSize = (int) (4 * ceil((float) (*width) / 4.0f)) * (*bytesPerPixel);
    //We are not interested in the padded bytes, so we allocate memory just for
    //the pixel data
    int unpaddedRowSize = (*width) * (*bytesPerPixel);
    //Total size of the pixel data in bytes
    int totalSize = unpaddedRowSize * (*height);
    *pixels = (uint8_t *) malloc(totalSize);
    //Read the pixel data Row by Row.
    //Data is padded and stored bottom-up
    int i = 0;
    //point to the last row of our pixel array (unpadded)
    uint8_t *currentRowPointer = *pixels + ((*height - 1) * unpaddedRowSize);
    for (i = 0; i < *height; i++) {
        //put file cursor in the next row from top to bottom
        fseek(imageFile, dataOffset + (i * paddedRowSize), SEEK_SET);
        //read only unpaddedRowSize bytes (we can ignore the padding bytes)
        fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
        //point to the next row (from bottom to top)
        currentRowPointer -= unpaddedRowSize;
    }

    fclose(imageFile);
}

//***Inputs*****
//fileName: The name of the file to save 
//pixels: Pointer to the pixel data array
//width: The width of the image in pixels
//height: The height of the image in pixels
//bytesPerPixel: The number of bytes per pixel that are used in the image
void WriteImage(const char *fileName, uint8_t *pixels, uint32_t width, uint32_t height, uint32_t bytesPerPixel) {
    //Open file in binary mode
    FILE *outputFile = fopen(fileName, "wb");
    //*****HEADER************//
    //write signature
    const char *BM = "BM";
    fwrite(&BM[0], 1, 1, outputFile);
    fwrite(&BM[1], 1, 1, outputFile);
    //Write file size considering padded bytes
    int paddedRowSize = (int) (4 * ceil((float) width / 4.0f)) * bytesPerPixel;
    uint32_t fileSize = paddedRowSize * height + HEADER_SIZE + INFO_HEADER_SIZE;
    fwrite(&fileSize, 4, 1, outputFile);
    //Write reserved
    uint32_t reserved = 0x0000;
    fwrite(&reserved, 4, 1, outputFile);
    //Write data offset
    uint32_t dataOffset = HEADER_SIZE + INFO_HEADER_SIZE;
    fwrite(&dataOffset, 4, 1, outputFile);

    //*******INFO*HEADER******//
    //Write size
    uint32_t infoHeaderSize = INFO_HEADER_SIZE;
    fwrite(&infoHeaderSize, 4, 1, outputFile);
    //Write width and height
    fwrite(&width, 4, 1, outputFile);
    fwrite(&height, 4, 1, outputFile);
    //Write planes
    int16_t planes = 1; //always 1
    fwrite(&planes, 2, 1, outputFile);
    //write bits per pixel
    int16_t bitsPerPixel = bytesPerPixel * 8;
    fwrite(&bitsPerPixel, 2, 1, outputFile);
    //write compression
    uint32_t compression = NO_COMPRESION;
    fwrite(&compression, 4, 1, outputFile);
    //write image size (in bytes)
    uint32_t imageSize = width * height * bytesPerPixel;
    fwrite(&imageSize, 4, 1, outputFile);
    //write resolution (in pixels per meter)
    uint32_t resolutionX = 11811; //300 dpi
    uint32_t resolutionY = 11811; //300 dpi
    fwrite(&resolutionX, 4, 1, outputFile);
    fwrite(&resolutionY, 4, 1, outputFile);
    //write colors used
    uint32_t colorsUsed = MAX_NUMBER_OF_COLORS;
    fwrite(&colorsUsed, 4, 1, outputFile);
    //Write important colors
    uint32_t importantColors = ALL_COLORS_REQUIRED;
    fwrite(&importantColors, 4, 1, outputFile);
    //write data
    int i = 0;
    int unpaddedRowSize = width * bytesPerPixel;
    for (i = 0; i < height; i++) {
        //start writing from the beginning of last row in the pixel array
        int pixelOffset = ((height - i) - 1) * unpaddedRowSize;
        fwrite(&pixels[pixelOffset], 1, paddedRowSize, outputFile);
    }
    fclose(outputFile);
}
