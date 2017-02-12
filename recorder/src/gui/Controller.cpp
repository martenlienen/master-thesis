#include <iostream>
#include <utility>

#include <boost/filesystem.hpp>
#include <wx/button.h>
#include <wx/msgdlg.h>
#include <wx/sizer.h>

#include "Controller.h"

namespace recorder {

namespace gui {

Controller::Controller(
    std::string subject, std::string directory,
    std::vector<std::tuple<std::string, std::string, std::string>> gestures,
    std::unique_ptr<agents::DVSAgent> dvs_agent,
    std::unique_ptr<agents::OpenCVAgent> cv_agent)
    : wxFrame(nullptr, wxID_ANY, "Controller"), recording(false),
      num_gestures(gestures.size()), current(0), subject(subject),
      directory(directory), gestures(gestures), dvs_agent(std::move(dvs_agent)),
      cv_agent(std::move(cv_agent)), dvs_frame(nullptr), cv_frame(nullptr) {
  this->counter_label = new wxStaticText(this, wxID_ANY, "", wxDefaultPosition,
                                         wxDefaultSize, wxALIGN_RIGHT);
  this->gesture_label = new wxStaticText(this, wxID_ANY, "");
  this->record_button = new wxButton(this, wxID_ANY, "Start");
  auto replay_button = new wxButton(this, wxID_ANY, "Replay");
  auto restart_button = new wxButton(this, wxID_ANY, "Restart DVS");
  auto prev_button = new wxButton(this, wxID_ANY, "<");
  auto next_button = new wxButton(this, wxID_ANY, ">");

  auto sizer = new wxBoxSizer(wxVERTICAL);
  auto top_sizer = new wxBoxSizer(wxHORIZONTAL);
  auto bottom_sizer = new wxBoxSizer(wxHORIZONTAL);
  top_sizer->Add(this->gesture_label, wxSizerFlags().Proportion(1));
  top_sizer->AddStretchSpacer();
  top_sizer->Add(this->counter_label);
  auto bottom_flags = wxSizerFlags().Center().Expand();
  bottom_sizer->Add(prev_button, bottom_flags.Proportion(0));
  bottom_sizer->Add(this->record_button, bottom_flags.Proportion(1));
  bottom_sizer->Add(replay_button, bottom_flags.Proportion(1));
  bottom_sizer->Add(restart_button, bottom_flags.Proportion(1));
  bottom_sizer->Add(next_button, bottom_flags.Proportion(0));
  sizer->Add(top_sizer, wxSizerFlags().Border(wxALL, 10));
  sizer->Add(bottom_sizer, wxSizerFlags().Border(wxALL, 10));

  this->SetSizerAndFit(sizer);

  // Set up event handling
  prev_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->current > 0) {
      this->current -= 1;
    }

    if (this->recording) {
      this->stopRecording();
    }

    this->updateLabels();
  });
  next_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->current < this->num_gestures - 1) {
      this->current += 1;
    }

    if (this->recording) {
      this->stopRecording();
    }

    this->updateLabels();
  });
  this->record_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->recording) {
      this->stopRecording();
      this->recording = false;
    } else {
      this->startRecording();
      this->recording = true;
    }

    this->updateLabels();
  });
  restart_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->recording) {
      this->stopRecording();
    }

    auto device = this->dvs_agent->device;
    boost::filesystem::path device_path(device);
    if (!boost::filesystem::exists(device_path)) {
      if (device[device.size() - 1] == '0') {
        device = "/dev/ttyUSB1";
      } else {
        device = "/dev/ttyUSB0";
      }
    }

    this->dvs_agent->stop();
    this->dvs_agent.reset(new agents::DVSAgent());
    this->dvs_agent->start(device);

    if (this->dvs_frame) {
      this->dvs_agent->setDisplay(this->dvs_frame->display);
    }
  });

  this->toggleDVSFrame();
  this->toggleOpenCVFrame();

  this->updateLabels();
}

void Controller::toggleDVSFrame() {
  if (this->dvs_frame) {
    this->dvs_frame->Close();
    this->dvs_frame = nullptr;
  } else {
    this->dvs_frame = new DVSFrame(this);
    this->dvs_frame->Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent &e) {
      this->dvs_agent->setDisplay(nullptr);
      e.Skip();
    });

    this->dvs_agent->setDisplay(this->dvs_frame->display);

    this->dvs_frame->Show();
  }
}

void Controller::toggleOpenCVFrame() {
  if (this->cv_frame) {
    this->cv_frame->Close();
    this->cv_frame = nullptr;
  } else {
    this->cv_frame = new OpenCVFrame(this);
    this->cv_frame->Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent &e) {
      this->cv_agent->setDisplay(nullptr);
      e.Skip();
    });

    this->cv_agent->setDisplay(this->cv_frame->display);

    this->cv_frame->Show();
  }
}

void Controller::startRecording() {
  if (this->num_gestures == 0) {
    return;
  }

  if (this->currentFileExists()) {
    auto msg = "Do you really want to overwrite this recording?";
    auto dlg = new wxMessageDialog(this, msg, "", wxYES_NO | wxCENTER);

    if (dlg->ShowModal() != wxID_YES) {
      return;
    }
  }

  auto id = std::get<1>(this->gestures[this->current]);
  boost::filesystem::path dir(this->directory);
  dir /= this->subject;

  boost::filesystem::create_directories(dir);

  this->dvs_agent->startRecording((dir / (id + ".aedat")).string());
  this->cv_agent->startRecording((dir / (id + ".mkv")).string());
}

void Controller::stopRecording() {
  this->dvs_agent->stopRecording();
  this->cv_agent->stopRecording();
}

void Controller::updateLabels() {
  if (this->num_gestures > 0) {
    auto label = std::to_string(this->current + 1) + " / " +
                 std::to_string(this->num_gestures);
    this->counter_label->SetLabel(label);

    auto name = std::get<0>(this->gestures[this->current]);

    if (this->currentFileExists()) {
      this->gesture_label->SetLabel(name + " (exists)");
    } else {
      this->gesture_label->SetLabel(name);
    }

    if (this->recording) {
      this->record_button->SetLabel("Stop Recording");
    } else {
      this->record_button->SetLabel("Start Recording");
    }
  }
}

bool Controller::currentFileExists() {
  if (this->num_gestures == 0) {
    return false;
  }

  auto id = std::get<1>(this->gestures[this->current]);
  auto path =
      boost::filesystem::path(this->directory) / this->subject / (id + ".mkv");

  return boost::filesystem::exists(path);
}
}
}
