#include "App.h"

namespace labeler {

bool App::OnInit() {
  this->main_frame = new MainFrame();
  this->main_frame->Show();

  return true;
}
}
