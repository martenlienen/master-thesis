#ifndef RECORDER_GUI_OPENCV_FRAME_H_
#define RECORDER_GUI_OPENCV_FRAME_H_

#include <wx/frame.h>

#include "OpenCVDisplay.h"

namespace recorder {

namespace gui {

class OpenCVFrame : public wxFrame {
public:
  OpenCVDisplay *display;

  OpenCVFrame(wxFrame *parent);
};
}
}

#endif // RECORDER_GUI_OPENCV_FRAME_H_
