#include <opencv2/opencv.hpp>
#include "saver.h"

void save_grayscale_image(const char *output_path, const std::vector<std::vector<uint8_t>> &image_buffer) {
    cv::Mat image((int) image_buffer.size(), (int) image_buffer[0].size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < image_buffer.size(); i++) {
        for (int j = 0; j < image_buffer[0].size(); j++) {
            image.data[i * image_buffer[0].size() + j] = image_buffer.at(i).at(j);
        }
    }
    cv::imwrite(output_path, image);
}