#include <iostream>

#include <wx/font.h>
#include <wx/sizer.h>
#include <wx/timer.h>

#include "Instructor.h"

namespace recorder {

namespace gui {

Instructor::Instructor(wxFrame *parent)
    : wxFrame(parent, wxID_ANY, "Instructor"),
      player(new wxMediaCtrl(this, wxID_ANY)),
      label(new wxStaticText(this, wxID_ANY, "", wxDefaultPosition,
                             wxDefaultSize, wxALIGN_CENTRE_HORIZONTAL)) {
  wxFont font(wxFontInfo(80).Bold());
  this->label->SetFont(font);

  auto sizer = new wxBoxSizer(wxVERTICAL);
  sizer->AddStretchSpacer();
  sizer->Add(this->player);
  sizer->Add(this->label, wxSizerFlags().Center());
  sizer->AddStretchSpacer();

  this->SetSizerAndFit(sizer);

  // After loading always start playing
  this->Bind(wxEVT_MEDIA_LOADED, [this](const wxMediaEvent &e) {
    this->label->Hide();
    this->player->Show();
    int w, h;
    this->GetSize(&w, &h);
    this->player->SetSize(0, 0, w, h);
    this->player->Play();
  });

  // Play video continuously
  this->Bind(wxEVT_MEDIA_STOP, [this](wxMediaEvent &e) {
    this->player->Seek(0);
    e.Veto();
  });
}

void Instructor::playInstructions(std::string path) {
  this->player->Load(path);
}

void Instructor::countdown(std::function<void()> callback) {
  this->time_left = 2;
  auto timer = new wxTimer(this);

  this->drawCountdown();

  this->Bind(wxEVT_TIMER, [this, timer, callback](wxTimerEvent &e) {
    this->time_left -= 1;

    this->drawCountdown();

    if (this->time_left == 0) {
      timer->Stop();
      callback();
    }
  });

  // Notify once per second
  timer->Start(1000.0);
}

void Instructor::drawCountdown() {
  this->label->SetLabel(std::to_string(this->time_left));
  this->label->Show();
  this->player->Hide();
}
}
}
