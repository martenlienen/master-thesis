#ifndef RECORDER_CAPTURE_DVS_CAPTURE_H_
#define RECORDER_CAPTURE_DVS_CAPTURE_H_

#include <atomic>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "edvs.h"
}

namespace recorder {

namespace capture {

class DVSCapture {
public:
  std::string uri;

  DVSCapture(std::string uri);
  ~DVSCapture();

  void start();
  void stop();

  std::vector<edvs_event_t> getEvents();

private:
  std::thread thread;
  std::atomic<bool> stopped;
  edvs_stream_t *stream_handle;
  std::mutex events_mutex;
  std::queue<edvs_event_t> events;

  void grabEvents();
};
}
}

#endif // RECORDER_CAPTURE_DVS_CAPTURE_H_
