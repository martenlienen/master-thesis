#include <chrono>

#include "OpenCVAgent.h"

namespace recorder {

namespace agents {

OpenCVAgent::OpenCVAgent(int rotate_degrees)
    : rotate_degrees(rotate_degrees), started(false), display(nullptr) {}

void OpenCVAgent::start(uint32_t camera_id) {
  this->started = true;

  // Capture OpenCV signal
  this->capture.reset(new capture::OpenCVCapture(camera_id));
  this->capture->start();

  // Start the thread
  this->thread = std::thread(&OpenCVAgent::run, this);
}

OpenCVAgent::~OpenCVAgent() { this->stop(); }

void OpenCVAgent::stop() {
  if (this->capture) {
    this->capture->stop();
  }

  this->stopRecording();
  this->started = false;

  if (this->thread.joinable()) {
    this->thread.join();
  }
}

void OpenCVAgent::setDisplay(gui::OpenCVDisplay *display) {
  std::lock_guard<std::mutex> guard(this->display_mutex);
  this->display = display;
}

void OpenCVAgent::startRecording(std::string path) {
  std::lock_guard<std::mutex> guard(this->storage_mutex);
  this->storage.reset(new store::VideoStorage(path));
}

void OpenCVAgent::stopRecording() {
  std::lock_guard<std::mutex> guard(this->storage_mutex);
  this->storage.reset();
}

void OpenCVAgent::run() {
  const auto ONE_MS = std::chrono::milliseconds(1);

  while (this->started) {
    // Collect frames
    auto frames = this->capture->getFrames();

    if (frames.size() > 0) {
      // Rotate frames
      if (this->rotate_degrees == 180) {
        for (auto &f : frames) {
          cv::flip(f, f, -1);
        }
      }

      // Push frames to view
      {
        std::lock_guard<std::mutex> guard(this->display_mutex);
        if (this->display) {
          this->display->setFrame(frames.back());
        }
      }

      // Write frames into file
      {
        std::lock_guard<std::mutex> guard(this->storage_mutex);
        if (this->storage) {
          for (auto &f : frames) {
            this->storage->write(f);
          }
        }
      }
    }

    std::this_thread::sleep_for(ONE_MS);
  }
}
}
}
