#ifndef RECORDER_AGENTS_DVS_AGENT_H_
#define RECORDER_AGENTS_DVS_AGENT_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "../capture/DVSCapture.h"
#include "../gui/DVSDisplay.h"
#include "../store/AedatStorage.h"
#include "TimestampFile.h"

namespace recorder {

namespace agents {

class DVSAgent {
public:
  std::string device;

  DVSAgent();
  ~DVSAgent();

  void start(std::string device);
  void stop();
  void setDisplay(gui::DVSDisplay *display);
  void startRecording(std::string path);
  void stopRecording();
  void startLongRecording(std::string path);
  void stopLongRecording();
  void startGesture(std::string name);
  void stopGesture();

private:
  uint64_t num_events_long;
  std::thread thread;
  std::mutex display_mutex;
  gui::DVSDisplay *display;
  std::mutex storage_mutex;
  std::unique_ptr<store::AedatStorage> storage;
  std::mutex long_storage_mutex;
  std::unique_ptr<store::AedatStorage> long_storage;
  std::string current_gesture;
  uint64_t current_gesture_start;
  std::unique_ptr<TimestampFile> long_timestamps;
  std::unique_ptr<capture::DVSCapture> capture;
  std::atomic<bool> started;

  void run();
};
}
}

#endif // RECORDER_AGENTS_DVS_AGENT_H_
