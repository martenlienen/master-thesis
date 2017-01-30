#include <iostream>

#include "Instructor.h"

namespace recorder {

namespace gui {

Instructor::Instructor()
    : wxFrame(NULL, wxID_ANY, "Instructor"),
      player(new wxMediaCtrl(this, wxID_ANY)) {
  this->player->Load("test.webm");

  this->Bind(wxEVT_MEDIA_LOADED,
             [this](const wxMediaEvent &e) { this->player->Play(); },
             this->player->GetId(), this->player->GetId());
}
}
}
