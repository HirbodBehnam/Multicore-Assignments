# Sobel Edge Detection
## Implementation
For implementing the sobel edge detection algorithm we start by reading the image in openCV and moving the data to GPU. Now, we have the image but in BRG color scheme. To convert it to grayscale image, we use the formula used in [OpenCV Doc](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#void%20cvtColor%28InputArray%20src,%20OutputArray%20dst,%20int%20code,%20int%20dstCn%29). The formula is:
$$
Grayscale = 0.114B + 0.587G + 0.299R
$$
Now after we convert the image to grayscale, we shall brighten our image. To do so, we use this formula:
$$
X = \alpha X + \beta
$$
where $\alpha$ and $\beta$ are parameters which are given to program.
Next, we shall do the real sobel algorithm. To do so, we simply move the kernels on top of each pixel of image. The code with does the following is the block below:
```cpp
/**
 * Extracts a window for convolution.
 * @param image The image to extract the window from
 * @param width The width of image
 * @param height The height of image
 * @param window [OUT] The extracted window
 */
__device__ void extract_window(const uint8_t *image, const struct coordinates cor, const int width, const int height,
                               int16_t window[3][3]) {
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            const struct coordinates current_cord{
                    .x = cor.x + i,
                    .y = cor.y + j,
            };
            int16_t result = 0;
            if (current_cord.x >= 0 && current_cord.y >= 0 && current_cord.x < width && current_cord.y < height) {
                // in bounds of image
                result = image[coordinates_to_index(current_cord, width)];
            }
            window[j + 1][i + 1] = result;
        }
}
```
After we extract a window, we shall calculate the convolution in respect to a kernel. To do so, I wrote a macro:
```cpp
#define CONVOLUTION(WINDOW, KERNEL, RESULT, THRESHOLD) do { \
    RESULT = 0;                                             \
    for (int x = 0; x < 3; x++){                            \
        for (int y = 0; y < 3; y++){                        \
            RESULT += WINDOW[x][y] * KERNEL[x][y];          \
        }                                                   \
    }                                                       \
    if (RESULT > THRESHOLD)                                 \
        RESULT = THRESHOLD;                                 \
    if (RESULT < 0)                                         \
        RESULT = 0;                                         \
} while(0)
```
The reason I wrote a macro instead of function is the fact that I used `__constant__` for storing kernels.

Next, we simply calculate $\sqrt{Gx^2 + Gy^2}$ and store it into an array. At long last, we copy the buffer from GPU to main memory and save it into a file.

## Optimizations
Two optimizations are done. The first one is to use `__constant__` to store kernels instead of just storing them in a buffer. Next is that each warp captures the cache locality of each of it's threads by moving horizontally on image.

## Benchmark
Running on Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz and GTX 1070
### 8K image
#### CPU
```
Running with 1 as alpha and 0as beta and 70 as threshold

real	0m4.576s
user	0m4.369s
sys     0m0.205s
```
#### GPU with 256 threads and 64 blocks
```
Running with 1 as alpha and 0as beta and 70 as threshold
Reading input...
Brightening input...

real    0m3.273s
user    0m2.740s
sys     0m0.334s

```
### 2K image
#### CPU
```
Running with 1 as alpha and 0as beta and 70 as threshold
Reading input...
Brightening input...

real	0m1.033s
user	0m0.997s
sys	    0m0.036s
```
#### GPU
```
Running with 1 as alpha and 0as beta and 70 as threshold

real	0m0.728s
user	0m0.630s
sys	    0m0.096s
```