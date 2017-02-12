#ifndef RECORDER_APP_H_
#define RECORDER_APP_H_

#include <wx/wx.h>

#include "gui/Setup.h"

namespace recorder {

class App : public wxApp {
public:
  virtual bool OnInit();

private:
  gui::Setup* setup;
};
}

#endif // RECORDER_APP_H_
