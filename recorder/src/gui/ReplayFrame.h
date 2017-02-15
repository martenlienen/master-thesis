#ifndef RECORDER_GUI_REPLAY_FRAME_H_
#define RECORDER_GUI_REPLAY_FRAME_H_

#include <string>

#include <wx/frame.h>
#include <wx/mediactrl.h>

namespace recorder {

namespace gui {

class ReplayFrame : public wxFrame {
public:
  ReplayFrame(wxFrame *parent, std::string path);

private:
  wxMediaCtrl *player;
};
}
}

#endif // RECORDER_GUI_REPLAY_FRAME_H_
