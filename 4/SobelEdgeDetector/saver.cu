#include <opencv2/opencv.hpp>
#include "saver.cuh"

void save_grayscale_image_gpu(const char *output_path, const uint8_t *image_buffer, int width, int height) {
    cv::Mat image(height, width, CV_8UC1, cv::Scalar(0));
    cudaMemcpy(image.data, image_buffer, width * height, cudaMemcpyDeviceToHost);
    cv::imwrite(output_path, image);
}