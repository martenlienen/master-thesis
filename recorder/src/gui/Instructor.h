#ifndef RECORDER_GUI_INSTRUCTOR_H_
#define RECORDER_GUI_INSTRUCTOR_H_

#include <wx/frame.h>
#include <wx/mediactrl.h>

namespace recorder {

namespace gui {

class Instructor : public wxFrame {
public:
  Instructor();

private:
  wxMediaCtrl *player;
};
}
}

#endif // RECORDER_GUI_INSTRUCTOR_H_
