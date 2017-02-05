#include <iostream>
#include <thread>

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

void OpenCVCapture::run() {
  this->stopped = false;
  std::thread t(&OpenCVCapture::grabFrames, this);
  t.detach();
}

void OpenCVCapture::stop() { this->stopped = true; }

void OpenCVCapture::grabFrames() {
  cv::VideoCapture cap(this->camera_id);

  while (!this->stopped.load()) {
    cv::Mat m;

    if (!cap.read(m)) {
      return;
    }

    {
      std::lock_guard<std::mutex> guard(this->last_frame_mutex);
      this->last_frame = m;
    }

    {
      std::lock_guard<std::mutex> guard(this->frames_mutex);
      this->frames.push(m);
    }
  }
}
}
}
