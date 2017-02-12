#ifndef RECORDER_GUI_CONTROLLER_H_
#define RECORDER_GUI_CONTROLLER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <wx/button.h>
#include <wx/frame.h>
#include <wx/stattext.h>

#include "../agents/DVSAgent.h"
#include "../agents/OpenCVAgent.h"
#include "DVSFrame.h"
#include "OpenCVFrame.h"

namespace recorder {

namespace gui {

class Controller : public wxFrame {
public:
  Controller(
      std::string subject, std::string directory,
      std::vector<std::tuple<std::string, std::string, std::string>> gestures,
      std::unique_ptr<agents::DVSAgent> dvs_agent,
      std::unique_ptr<agents::OpenCVAgent> cv_agent);

private:
  int num_gestures;
  int current;
  bool recording;
  std::string subject;
  std::string directory;
  std::vector<std::tuple<std::string, std::string, std::string>> gestures;
  std::unique_ptr<agents::DVSAgent> dvs_agent;
  std::unique_ptr<agents::OpenCVAgent> cv_agent;
  gui::DVSFrame *dvs_frame;
  gui::OpenCVFrame *cv_frame;

  wxStaticText* counter_label;
  wxStaticText* gesture_label;
  wxButton* record_button;

  void toggleDVSFrame();
  void toggleOpenCVFrame();

  void startRecording();
  void stopRecording();

  void updateLabels();

  bool currentFileExists();
};
}
}

#endif // RECORDER_GUI_CONTROLLER_H_
