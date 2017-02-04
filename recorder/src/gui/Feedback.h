#ifndef RECORDER_GUI_FEEDBACK_H_
#define RECORDER_GUI_FEEDBACK_H_

#include <wx/frame.h>
#include <wx/timer.h>

#include "../capture/OpenCVCapture.h"

namespace recorder {

namespace gui {

class Feedback : public wxFrame {
public:
  Feedback(capture::OpenCVCapture &cv_capture);

private:
  capture::OpenCVCapture &cv_capture;
  wxTimer repaint_timer;

  void onPaint();
};
}
}

#endif // RECORDER_GUI_FEEDBACK_H_
