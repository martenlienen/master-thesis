#include "App.h"

namespace recorder {

bool App::OnInit() {
  this->controller = new gui::Controller();
  this->controller->Show();

  return true;
}
}
