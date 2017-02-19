#ifndef RECORDER_GUI_INSTRUCTOR_H_
#define RECORDER_GUI_INSTRUCTOR_H_

#include <functional>
#include <string>

#include <wx/frame.h>
#include <wx/mediactrl.h>
#include <wx/stattext.h>

namespace recorder {

namespace gui {

class Instructor : public wxFrame {
public:
  Instructor(wxFrame *parent);

  void playInstructions(std::string path);
  void countdown(std::function<void()> callback);

private:
  uint8_t time_left;
  wxMediaCtrl *player;
  wxStaticText *label;

  void drawCountdown();
};
}
}

#endif // RECORDER_GUI_INSTRUCTOR_H_
