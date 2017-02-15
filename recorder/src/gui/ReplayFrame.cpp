#include "ReplayFrame.h"

namespace recorder {

namespace gui {

ReplayFrame::ReplayFrame(wxFrame *parent, std::string path)
    : wxFrame(parent, wxID_ANY, "Replay"),
      player(new wxMediaCtrl(this, wxID_ANY)) {
  this->player->Load(path);
  this->player->ShowPlayerControls();

  this->Bind(wxEVT_MEDIA_LOADED,
             [this](const wxMediaEvent &e) { this->player->Play(); });

  // Play video continuously
  this->Bind(wxEVT_MEDIA_STOP, [this](wxMediaEvent &e) {
    this->player->Seek(0);
    e.Veto();
  });
}
}
}
