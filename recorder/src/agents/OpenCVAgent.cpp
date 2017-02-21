#include <chrono>

#include "OpenCVAgent.h"

namespace recorder {

namespace agents {

OpenCVAgent::OpenCVAgent(int rotate_degrees)
    : rotate_degrees(rotate_degrees), num_frames_long(0), started(false),
      display(nullptr) {}

void OpenCVAgent::start(uint32_t camera_id) {
  this->started = true;

  // Capture OpenCV signal
  this->capture.reset(new capture::OpenCVCapture(camera_id, FPS));
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
  this->stopLongRecording();
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
  this->storage.reset(new store::VideoStorage(path, FPS));
}

void OpenCVAgent::stopRecording() {
  std::lock_guard<std::mutex> guard(this->storage_mutex);
  this->storage.reset();
}

void OpenCVAgent::startLongRecording(std::string path) {
  std::lock_guard<std::mutex> guard(this->long_storage_mutex);
  this->long_storage.reset(new store::VideoStorage(path, FPS));
  this->num_frames_long = 0;
  this->long_timestamps.reset(new TimestampFile(path + ".csv"));
}

void OpenCVAgent::stopLongRecording() {
  std::lock_guard<std::mutex> guard(this->long_storage_mutex);
  this->long_storage.reset();
  this->long_timestamps.reset();
}

void OpenCVAgent::startGesture(std::string name) {
  std::lock_guard<std::mutex> guard(this->long_storage_mutex);
  this->current_gesture_start = this->num_frames_long;
  this->current_gesture = name;
}

void OpenCVAgent::stopGesture() {
  if (!this->long_timestamps) {
    return;
  }

  std::lock_guard<std::mutex> guard(this->long_storage_mutex);
  this->long_timestamps->pushTimestamp(this->current_gesture,
                                       this->current_gesture_start,
                                       this->num_frames_long);
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

      // Write frames into long file
      {
        std::lock_guard<std::mutex> guard(this->long_storage_mutex);
        this->num_frames_long += frames.size();
        if (this->long_storage) {
          for (auto &f : frames) {
            this->long_storage->write(f);
          }
        }
      }
    }

    std::this_thread::sleep_for(ONE_MS);
  }
}
}
}
