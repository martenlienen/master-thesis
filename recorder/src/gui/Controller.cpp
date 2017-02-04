#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <wx/button.h>
#include <wx/choice.h>
#include <wx/sizer.h>
#include <wx/stattext.h>

#include "Controller.h"

namespace recorder {

namespace gui {

Controller::Controller() : wxFrame(NULL, wxID_ANY, "Controller") {
  auto num_cameras = capture::OpenCVCapture::getNumCameras();
  auto camera_choices = wxArrayString();
  for (auto i = 0; i < num_cameras; i++) {
    camera_choices.Add(std::to_string(i));
  }

  // Create controls
  auto camera_label = new wxStaticText(this, wxID_ANY, "Camera");
  auto camera_choice = new wxChoice(this, wxID_ANY, wxDefaultPosition,
                                    wxDefaultSize, camera_choices);
  auto go_btn = new wxButton(this, wxID_ANY, "Go");
  camera_choice->SetSelection(0);

  // Arrange controls
  auto top_sizer = new wxBoxSizer(wxVERTICAL);
  auto form_sizer = new wxGridSizer(2, 10, 10);
  form_sizer->Add(camera_label, wxSizerFlags().Center());
  form_sizer->Add(camera_choice, wxSizerFlags().Center());
  top_sizer->Add(form_sizer, 1, wxEXPAND | wxALL, 10);
  top_sizer->Add(go_btn, 0, wxEXPAND | wxALL, 10);

  this->SetSizerAndFit(top_sizer);

  // Set up event handling
  camera_choice->Bind(wxEVT_CHOICE, [this](const wxCommandEvent &e) {
    this->camera_id = e.GetInt();
  });
  go_btn->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    this->startRecording(this->camera_id);
  });
}

void Controller::startRecording(uint32_t camera_id) {
  if (recording) {
    this->cv_capture->stop();
    this->feedback = nullptr;
    this->recording = false;
  } else {
    this->cv_capture.reset(new capture::OpenCVCapture(camera_id));
    this->feedback = new Feedback(*this->cv_capture);
    this->feedback->Show();
    this->cv_capture->run();
    this->recording = true;
  }
}
}
}
