#include "OpenCVFrame.h"

namespace recorder {

namespace gui {

OpenCVFrame::OpenCVFrame(wxFrame *parent)
    : wxFrame(parent, wxID_ANY, "OpenCVFrame"),
      display(new OpenCVDisplay(this, wxID_ANY)) {}
}
}
