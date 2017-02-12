#include "DVSFrame.h"

namespace recorder {

namespace gui {

DVSFrame::DVSFrame(wxFrame *parent)
    : wxFrame(parent, wxID_ANY, "DVS"),
      display(new DVSDisplay(this, wxID_ANY)) {}
}
}
