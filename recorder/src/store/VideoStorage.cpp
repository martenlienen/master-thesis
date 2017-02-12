#include "VideoStorage.h"

namespace recorder {

namespace store {

VideoStorage::VideoStorage(std::string path)
    : writer(path, cv::VideoWriter::fourcc('X', '2', '6', '4'), 30,
             cv::Size(640, 480)) {}

void VideoStorage::write(cv::Mat frame) { this->writer << frame; }
}
}
