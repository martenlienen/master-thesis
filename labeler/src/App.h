#ifndef LABELER_APP_H_
#define LABELER_APP_H_

#include <wx/wx.h>

#include "MainFrame.h"

namespace labeler {

class App : public wxApp {
public:
  virtual bool OnInit();

private:
  MainFrame* main_frame;
};
}

#endif // LABELER_APP_H_
