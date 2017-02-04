#ifndef RECORDER_CAPTURE_OPEN_CV_CAPTURE_H_
#define RECORDER_CAPTURE_OPEN_CV_CAPTURE_H_

#include <atomic>
#include <cstdint>
#include <mutex>

#include <opencv2/core/mat.hpp>

namespace recorder {

namespace capture {

class OpenCVCapture {
public:
  uint32_t camera_id;
  std::atomic<bool> stopped{false};
  std::mutex last_frame_mutex;
  cv::Mat last_frame;

  static uint32_t getNumCameras();

  OpenCVCapture(uint32_t camera_id);
  void run();
  void stop();

private:
  void grabFrames();
};
}
}

#endif // RECORDER_CAPTURE_OPEN_CV_CAPTURE_H_
