#ifndef RECORDER_GUI_DVS_FRAME_H_
#define RECORDER_GUI_DVS_FRAME_H_

#include <wx/frame.h>
#include <wx/timer.h>

#include "DVSDisplay.h"

namespace recorder {

namespace gui {

class DVSFrame : public wxFrame {
public:
  DVSDisplay *display;

  DVSFrame(wxFrame *parent);
};
}
}

#endif // RECORDER_GUI_DVS_FRAME_H_
