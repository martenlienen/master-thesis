#include <iostream>
#include <stdexcept>
#include <thread>

#include "DVSCapture.h"

namespace recorder {

namespace capture {

DVSCapture::DVSCapture(std::string uri) : uri(uri) {
  this->stream_handle = edvs_open(uri.c_str());

  if (!this->stream_handle) {
    throw std::runtime_error("Could not open device");
  }

  if (edvs_run(this->stream_handle) != 0) {
    edvs_close(this->stream_handle);
    throw std::runtime_error("Could not start device");
  }
}

DVSCapture::~DVSCapture() {
  this->stop();
  edvs_close(this->stream_handle);
}

void DVSCapture::start() {
  this->stopped = false;
  this->thread = std::thread(&DVSCapture::grabEvents, this);
}

void DVSCapture::stop() {
  this->stopped = true;

  if (this->thread.joinable()) {
    this->thread.join();
  }
}

std::vector<edvs_event_t> DVSCapture::getEvents() {
  std::lock_guard<std::mutex> guard(this->events_mutex);
  const int n = this->events.size();
  std::vector<edvs_event_t> events(n);

  for (int i = 0; i < n; i++) {
    events[n - i - 1] = this->events.front();
    this->events.pop();
  }

  return events;
}

void DVSCapture::grabEvents() {
  const uint16_t X_MAX = 127;
  const size_t EVENT_BUFFER_SIZE = 1024;
  edvs_event_t event_buffer[EVENT_BUFFER_SIZE];

  while (!this->stopped) {
    ssize_t n = edvs_read(this->stream_handle, event_buffer, EVENT_BUFFER_SIZE);

    if (n < 0) {
      std::cout << "Error while reading events from DVS device" << std::endl;
      break;
    }

    std::lock_guard<std::mutex> guard(this->events_mutex);
    for (ssize_t i = 0; i < n; i++) {
      // Mirror events along the x-axis so that they are stored the same way as
      // the camera frames, i.e. as if the DVS was your eye.
      event_buffer[i].x = X_MAX - event_buffer[i].x;

      // The events are in the buffer in reverse order (newest at the front)
      this->events.push(event_buffer[i]);
    }

    std::this_thread::yield();
  }
}
}
}
