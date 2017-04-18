#include "App.h"

namespace jaerrec {

bool App::OnInit() {
  this->main_frame = new MainFrame();
  this->main_frame->Show();

  return true;
}

int App::FilterEvent(wxEvent &e) {
  // Use this to define a global key event for going to the next gesture
  if (this->main_frame) {
    if (e.GetEventType() == wxEVT_KEY_UP) {
      auto key_event = (wxKeyEvent &)e;

      if (key_event.GetUnicodeKey() == ' ') {
        this->main_frame->nextGesture();

        return 1;
      }
    }
  }

  return wxApp::FilterEvent(e);
}
}
