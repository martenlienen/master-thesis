#include "App.h"

namespace recorder {

bool App::OnInit() {
  this->setup = new gui::Setup();
  this->setup->Show();

  return true;
}
}
