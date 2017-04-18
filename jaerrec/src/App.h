#ifndef JAERREC_APP_H_
#define JAERREC_APP_H_

#include <wx/wx.h>

#include "MainFrame.h"

namespace jaerrec {

class App : public wxApp {
public:
  virtual bool OnInit();
  virtual int FilterEvent(wxEvent &e);

private:
  MainFrame *main_frame;
};
}

#endif // JAERREC_APP_H_
