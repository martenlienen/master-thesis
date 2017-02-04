#ifndef RECORDER_APP_H_
#define RECORDER_APP_H_

#include <wx/wx.h>

#include "gui/Controller.h"

namespace recorder {

class App : public wxApp {
public:
  virtual bool OnInit();

private:
  gui::Controller* controller;
};
}

#endif // RECORDER_APP_H_
