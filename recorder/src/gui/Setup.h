#ifndef RECORDER_GUI_SETUP_H_
#define RECORDER_GUI_SETUP_H_

#include <utility>
#include <vector>

#include <wx/frame.h>

namespace recorder {

namespace gui {

class Setup : public wxFrame {
public:
  Setup();

private:
  std::string subject = "";
  uint32_t camera_id = 0;
  std::string dvs_device = "/dev/ttyUSB0";
  std::string directory;
  std::string gestures =
      "/home/cqql/projects/tum-thesis/instructions/gestures.csv";

  void startController();
  std::vector<std::tuple<std::string, std::string, std::string>>
  parseGestures();
};
}
}

#endif // RECORDER_GUI_SETUP_H_
