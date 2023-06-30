#include <opencv2/opencv.hpp>
#include "saver.cuh"

void save_grayscale_image_gpu(const char *output_path, const uint8_t *image_buffer, int width, int height) {
    cv::Mat image(height, width, CV_8UC1, cv::Scalar(0));
    cudaMemcpy(image.data, image_buffer, width * height, cudaMemcpyDeviceToHost);
    cv::imwrite(output_path, image);
}

void show_grayscale_image(const std::vector<std::vector<uint8_t>> &image_buffer) {
    cv::Mat image((int) image_buffer.size(), (int) image_buffer[0].size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < image_buffer.size(); i++) {
        for (int j = 0; j < image_buffer[0].size(); j++) {
            image.data[i * image_buffer[0].size() + j] = image_buffer.at(i).at(j);
        }
    }
    cv::imshow("image", image);
    cv::waitKey(0);
}