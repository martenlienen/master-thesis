#include <opencv2/videoio.hpp>

#include "OpenCVCapture.h"

namespace recorder {

namespace capture {

uint32_t OpenCVCapture::getNumCameras() {
  cv::VideoCapture cap;

  for (uint32_t i = 0; i < UINT32_MAX; i++) {
    bool exists = cap.open(i);
    cap.release();

    if (!exists) {
      return i;
    }
  }

  return UINT32_MAX;
}

OpenCVCapture::OpenCVCapture(uint32_t camera_id) : camera_id(camera_id) {}

OpenCVCapture::~OpenCVCapture() { this->stop(); }

void OpenCVCapture::start() {
  this->stopped = false;
  this->thread = std::thread(&OpenCVCapture::grabFrames, this);
}

void OpenCVCapture::stop() {
  this->stopped = true;

  if (this->thread.joinable()) {
    this->thread.join();
  }
}

std::vector<cv::Mat> OpenCVCapture::getFrames() {
  std::lock_guard<std::mutex> guard(this->frames_mutex);
  int n = this->frames.size();
  std::vector<cv::Mat> frames(n);

  for (int i = 0; i < n; i++) {
    frames[n - i - 1] = this->frames.front();
    this->frames.pop();
  }

  return frames;
}

void OpenCVCapture::grabFrames() {
  cv::VideoCapture cap(this->camera_id);

  while (!this->stopped.load()) {
    cv::Mat m;

    if (!cap.read(m)) {
      return;
    }

    std::lock_guard<std::mutex> guard(this->frames_mutex);
    this->frames.push(m);

    std::this_thread::yield();
  }
}
}
}
