#include <opencv2/opencv.hpp>
#include "reader.h"

std::vector<std::vector<uint8_t>> read_to_grayscale(const char *filepath) {
    // Load the image
    cv::Mat image = cv::imread(filepath);
    if (image.empty()) {
        std::cout << "Cannot load image." << std::endl;
        exit(1);
    }
    // Convert to grayscale
    std::vector<std::vector<uint8_t>> result(image.size().height, std::vector<uint8_t>(image.size().width));
    for (int i = 0; i < result.size(); i++)
        for (int j = 0; j < result[0].size(); j++) {
            cv::Vec3b bgrPixel = image.at<cv::Vec3b>(i, j);
            float sum = 0;
            sum += static_cast<float>(bgrPixel.val[0]) * 0.114f; // B
            sum += static_cast<float>(bgrPixel.val[1]) * 0.587f; // G
            sum += static_cast<float>(bgrPixel.val[2]) * 0.299f; // R
            result.at(i).at(j) = static_cast<uint8_t>(sum);
        }
    return result;
}