#ifndef RECORDER_GUI_CONTROLLER_H_
#define RECORDER_GUI_CONTROLLER_H_

#include <memory>

#include <wx/frame.h>

#include "../capture/OpenCVCapture.h"
#include "Feedback.h"

namespace recorder {

namespace gui {

class Controller : public wxFrame {
public:
  Controller();

private:
  uint32_t camera_id = 0;
  std::string path;
  bool recording = false;
  std::unique_ptr<recorder::capture::OpenCVCapture> cv_capture;
  gui::Feedback *feedback;

  void startRecording(uint32_t camera_id);
};
}
}

#endif // RECORDER_GUI_CONTROLLER_H_
