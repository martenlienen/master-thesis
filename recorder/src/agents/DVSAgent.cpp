#include <chrono>

#include "DVSAgent.h"

namespace recorder {

namespace agents {

DVSAgent::DVSAgent() : started(false), display(nullptr) {}

void DVSAgent::start(std::string device) {
  this->device = device;
  auto uri = device + "?baudrate=12000000&dtsm=3&htsm=2";
  this->started = true;

  // Capture DVS signal
  this->capture.reset(new capture::DVSCapture(uri));
  this->capture->start();

  // Start the thread
  this->thread = std::thread(&DVSAgent::run, this);
}

DVSAgent::~DVSAgent() { this->stop(); }

void DVSAgent::stop() {
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

void DVSAgent::setDisplay(gui::DVSDisplay *display) {
  std::lock_guard<std::mutex> guard(this->display_mutex);
  this->display = display;
}

void DVSAgent::startRecording(std::string path) {
  std::lock_guard<std::mutex> guard(this->storage_mutex);
  this->storage.reset(new store::AedatStorage(path));
}

void DVSAgent::stopRecording() {
  std::lock_guard<std::mutex> guard(this->storage_mutex);
  this->storage.reset();
}

void DVSAgent::startLongRecording(std::string path) {
  std::lock_guard<std::mutex> guard(this->long_storage_mutex);
  this->long_storage.reset(new store::AedatStorage(path));
}

void DVSAgent::stopLongRecording() {
  std::lock_guard<std::mutex> guard(this->long_storage_mutex);
  this->long_storage.reset();
}

void DVSAgent::run() {
  const auto ONE_MS = std::chrono::milliseconds(1);

  while (this->started) {
    // Collect events
    auto events = this->capture->getEvents();

    // Push events to view
    {
      std::lock_guard<std::mutex> guard(this->display_mutex);
      if (this->display) {
        this->display->pushEvents(events);
      }
    }

    // Write events into file
    {
      std::lock_guard<std::mutex> guard(this->storage_mutex);
      if (this->storage) {
        for (int i = events.size() - 1; i >= 0; i--) {
          this->storage->write(events[i]);
        }
      }
    }

    // Write events into long file
    {
      std::lock_guard<std::mutex> guard(this->long_storage_mutex);
      if (this->long_storage) {
        for (int i = events.size() - 1; i >= 0; i--) {
          this->long_storage->write(events[i]);
        }
      }
    }

    std::this_thread::sleep_for(ONE_MS);
  }
}
}
}
