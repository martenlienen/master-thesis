#ifndef RECORDER_STORE_VIDEO_STORAGE_H_
#define RECORDER_STORE_VIDEO_STORAGE_H_

#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

namespace recorder {

namespace store {

class VideoStorage {
public:
  VideoStorage(std::string path);

  void write(cv::Mat frame);

private:
  cv::VideoWriter writer;
};
}
}

#endif // RECORDER_STORE_VIDEO_STORAGE_H_
