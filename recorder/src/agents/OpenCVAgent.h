#ifndef RECORDER_AGENTS_OPEN_CV_AGENT_H_
#define RECORDER_AGENTS_OPEN_CV_AGENT_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "../capture/OpenCVCapture.h"
#include "../gui/OpenCVDisplay.h"
#include "../store/VideoStorage.h"

namespace recorder {

namespace agents {

class OpenCVAgent {
public:
  OpenCVAgent(int rotate_degrees);
  ~OpenCVAgent();

  void start(uint32_t camera_id);
  void stop();
  void setDisplay(gui::OpenCVDisplay *display);
  void startRecording(std::string path);
  void stopRecording();
  void startLongRecording(std::string path);
  void stopLongRecording();

private:
  int rotate_degrees;
  std::thread thread;
  std::mutex display_mutex;
  gui::OpenCVDisplay *display;
  std::mutex storage_mutex;
  std::unique_ptr<store::VideoStorage> storage;
  std::mutex long_storage_mutex;
  std::unique_ptr<store::VideoStorage> long_storage;
  std::unique_ptr<capture::OpenCVCapture> capture;
  std::atomic<bool> started;

  void run();
};
}
}

#endif // RECORDER_AGENTS_OPEN_CV_AGENT_H_
