#ifndef RECORDER_CAPTURE_OPEN_CV_CAPTURE_H_
#define RECORDER_CAPTURE_OPEN_CV_CAPTURE_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace recorder {

namespace capture {

class OpenCVCapture {
public:
  uint32_t camera_id;
  const uint32_t fps;

  static uint32_t getNumCameras();

  OpenCVCapture(uint32_t camera_id, uint32_t fps);
  ~OpenCVCapture();
  void start();
  void stop();
  std::vector<cv::Mat> getFrames();

private:
  std::thread thread;
  std::atomic<bool> stopped{false};
  std::mutex frames_mutex;
  std::queue<cv::Mat> frames;

  void grabFrames();
};
}
}

#endif // RECORDER_CAPTURE_OPEN_CV_CAPTURE_H_
