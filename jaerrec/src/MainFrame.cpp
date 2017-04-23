#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>

#include <boost/filesystem.hpp>
#include <wx/button.h>
#include <wx/menu.h>
#include <wx/sizer.h>

#include "ConnectDialog.h"
#include "MainFrame.h"

namespace jaerrec {

MainFrame::MainFrame()
    : wxFrame(NULL, wxID_ANY, "Record with jAER"),
      player(new wxMediaCtrl(this, wxID_ANY)),
      header(new wxStaticText(this, wxID_ANY, "Not Recording",
                              wxDefaultPosition, wxDefaultSize,
                              wxALIGN_CENTER_VERTICAL)),
      stop_button(new wxButton(this, wxID_ANY, "Stop Recording")) {
  auto font = this->header->GetFont();
  this->header->SetFont(font.MakeLarger().MakeLarger().MakeLarger().MakeBold());

  this->stop_button->Disable();

  auto main_sizer = new wxBoxSizer(wxVERTICAL);
  main_sizer->Add(this->header, wxSizerFlags().Center().Border(wxALL, 10));
  main_sizer->Add(this->player, wxSizerFlags().Expand().Proportion(1));
  main_sizer->Add(this->stop_button, wxSizerFlags().Expand());

  this->SetSizerAndFit(main_sizer);

  // Set up the menu bar
  auto menu_bar = new wxMenuBar();
  auto file_menu = new wxMenu();
  file_menu->Append(wxID_OPEN, "&Open");
  menu_bar->Append(file_menu, "&File");
  this->SetMenuBar(menu_bar);

  this->Bind(wxEVT_MENU,
             [this](const wxCommandEvent &e) {
               auto dlg = new ConnectDialog(this);

               if (dlg->ShowModal() == wxID_OK) {
                 // If we are already recording
                 if (this->curr_gesture >= 0) {
                   this->stopRecording();
                 }

                 this->subject = dlg->subject;
                 this->ip = dlg->ip;
                 this->port = dlg->port;
                 this->instruction_dir = dlg->instruction_dir;
                 this->logging_dir = dlg->logging_dir;
                 this->gestures = dlg->gestures;

                 this->startRecording();
               }
             },
             wxID_OPEN, wxID_OPEN);

  this->Bind(wxEVT_BUTTON,
             [this](const wxCommandEvent &e) { this->stopRecording(); });

  // Play videos once they are loaded
  this->player->Bind(wxEVT_MEDIA_LOADED,
                     [this](const wxMediaEvent &e) { this->player->Play(); });

  // Play video continuously
  this->player->Bind(wxEVT_MEDIA_STOP, [this](wxMediaEvent &e) {
    this->player->Seek(0);
    e.Veto();
  });

  // Stop a maybe on-going recording before closing the application
  this->Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent &e) {
    if (this->curr_gesture >= 0) {
      this->stopRecording();
    }

    // Do not veto closing the window
    e.Skip();
  });
}

void MainFrame::startRecording() {
  // Instruct jAER to start logging into a file
  auto now = std::time(nullptr);
  auto tm = *std::localtime(&now);
  std::stringstream filename;
  filename << this->logging_dir << this->subject << "-"
           << std::put_time(&tm, "%Y%m%d-%H%M%S") << ".aedat";
  this->sendCommand("startlogging " + filename.str());

  this->curr_gesture = 0;
  this->updateView();
}

void MainFrame::stopRecording() {
  // Stop jAER logging
  this->sendCommand("stoplogging");

  this->curr_gesture = -1;
  this->gestures.clear();
  this->updateView();
}

void MainFrame::sendCommand(const std::string command) const {
  std::stringstream shell_cmd;
  shell_cmd << "echo '" << command << "' | ncat --udp " << this->ip << " "
            << this->port;
  std::system(shell_cmd.str().c_str());
}

void MainFrame::nextGesture() {
  if (this->curr_gesture >= 0) {
    this->curr_gesture += 1;
    this->updateView();
  }
}

void MainFrame::updateView() {
  if (this->curr_gesture < 0) {
    this->header->SetLabel("Not Recording");
    this->stop_button->Disable();
    this->player->Stop();
  } else if (this->curr_gesture >= this->gestures.size()) {
    this->header->SetLabel("Done");
    this->stop_button->Enable();
    this->player->Stop();
  } else {
    auto name = this->gestures[this->curr_gesture];

    this->header->SetLabel(std::string("Recording ") + name + " (" +
                           std::to_string(this->curr_gesture + 1) + ")");
    this->stop_button->Enable();

    const auto vid_path =
        boost::filesystem::path(this->instruction_dir) / (name + ".mkv");

    if (boost::filesystem::exists(vid_path)) {
      // Stop the player before loading. Somewhere in wxMediaCtrl is a
      // race-condition that can segfaults if you load while a video is playing.
      this->player->Stop();

      this->player->Load(vid_path.string());
    }
  }
}
}
