#include "App.h"

#include "gui/Instructor.h"

namespace recorder {

bool App::OnInit() {
  this->controller.reset(new wxFrame(NULL, wxID_ANY, "Controller"));
  this->instructor.reset(new gui::Instructor());
  this->feedback.reset(new wxFrame(NULL, wxID_ANY, "Feedback"));

  this->controller->Show();
  this->instructor->Show();
  this->feedback->Show();

  return true;
}
}
