#include <algorithm>
#include <iostream>
#include <utility>

#include <boost/filesystem.hpp>
#include <wx/button.h>
#include <wx/msgdlg.h>
#include <wx/sizer.h>

#include "Controller.h"
#include "ReplayFrame.h"

namespace recorder {

namespace gui {

Controller::Controller(
    std::string subject, std::string directory,
    std::vector<std::tuple<std::string, std::string, std::string>> gestures,
    std::unique_ptr<agents::DVSAgent> dvs_agent,
    std::unique_ptr<agents::OpenCVAgent> cv_agent)
    : wxFrame(nullptr, wxID_ANY, "Controller"), recording(false),
      long_recording(false), num_gestures(gestures.size()), current(0),
      subject(subject), directory(directory), gestures(gestures),
      dvs_agent(std::move(dvs_agent)), cv_agent(std::move(cv_agent)),
      dvs_frame(nullptr), cv_frame(nullptr) {
  this->instructor = new Instructor(this);
  this->instructor->Show();

  this->counter_label = new wxStaticText(this, wxID_ANY, "", wxDefaultPosition,
                                         wxDefaultSize, wxALIGN_RIGHT);
  this->gesture_label = new wxStaticText(this, wxID_ANY, "");
  this->record_button = new wxButton(this, wxID_ANY, "Start");
  this->long_record_button =
      new wxButton(this, wxID_ANY, "Start long recording");
  this->replay_button = new wxButton(this, wxID_ANY, "Replay");
  auto restart_button = new wxButton(this, wxID_ANY, "Restart DVS");
  auto prev_button = new wxButton(this, wxID_ANY, "<");
  auto next_button = new wxButton(this, wxID_ANY, ">");
  auto toggle_cv_button = new wxButton(this, wxID_ANY, "Toggle camera view");
  auto toggle_dvs_button = new wxButton(this, wxID_ANY, "Toggle DVS view");
  auto randomize_button = new wxButton(this, wxID_ANY, "Randomize Order");

  auto sizer = new wxBoxSizer(wxVERTICAL);
  auto top_sizer = new wxBoxSizer(wxHORIZONTAL);
  auto bottom_sizer = new wxBoxSizer(wxHORIZONTAL);
  auto toggle_sizer = new wxBoxSizer(wxHORIZONTAL);
  top_sizer->Add(this->gesture_label, wxSizerFlags().Proportion(1));
  top_sizer->AddStretchSpacer();
  top_sizer->Add(this->counter_label);
  auto bottom_flags = wxSizerFlags().Center().Expand();
  bottom_sizer->Add(prev_button, bottom_flags.Proportion(0));
  bottom_sizer->Add(this->record_button, bottom_flags.Proportion(1));
  bottom_sizer->Add(this->replay_button, bottom_flags.Proportion(1));
  bottom_sizer->Add(restart_button, bottom_flags.Proportion(1));
  bottom_sizer->Add(next_button, bottom_flags.Proportion(0));
  toggle_sizer->Add(toggle_cv_button, bottom_flags.Proportion(1));
  toggle_sizer->Add(toggle_dvs_button, bottom_flags.Proportion(1));
  toggle_sizer->Add(this->long_record_button, bottom_flags.Proportion(1));
  toggle_sizer->Add(randomize_button, bottom_flags.Proportion(1));
  sizer->Add(top_sizer, wxSizerFlags().Border(wxALL, 10));
  sizer->Add(bottom_sizer, wxSizerFlags().Border(wxALL, 10));
  sizer->Add(toggle_sizer, wxSizerFlags().Border(wxALL, 10));

  this->SetSizerAndFit(sizer);

  // Set up event handling
  prev_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->current > 0) {
      this->current -= 1;
      this->playCurrentInstruction();
    }

    if (this->recording) {
      this->stopRecording();
    }

    this->updateLabels();
  });
  next_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->current < this->num_gestures - 1) {
      this->current += 1;
      this->playCurrentInstruction();
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
  this->long_record_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->long_recording) {
      this->stopLongRecording();
      this->long_recording = false;
    } else {
      this->startLongRecording();
      this->long_recording = true;
    }

    this->updateLabels();
  });
  this->replay_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->recording) {
      this->stopRecording();
    }

    auto id = std::get<1>(this->gestures[this->current]);
    auto path = boost::filesystem::path(this->directory) / this->subject /
                (id + ".mkv");
    auto replay_frame = new ReplayFrame(this, path.string());
    replay_frame->Show();
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
  toggle_cv_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    this->toggleOpenCVFrame();
  });
  toggle_dvs_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    this->toggleDVSFrame();
  });
  randomize_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (recording) {
      this->stopRecording();
    }

    std::random_shuffle(this->gestures.begin(), this->gestures.end());

    this->updateLabels();
  });

  this->toggleDVSFrame();
  this->toggleOpenCVFrame();

  this->updateLabels();

  this->playCurrentInstruction();
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

  // Ensure that directory exists
  boost::filesystem::create_directories(dir);

  this->dvs_agent->startRecording((dir / (id + ".aedat")).string());
  this->cv_agent->startRecording((dir / (id + ".mkv")).string());

  this->dvs_agent->startGesture(id);
  this->cv_agent->startGesture(id);
}

void Controller::stopRecording() {
  this->dvs_agent->stopRecording();
  this->cv_agent->stopRecording();

  this->dvs_agent->stopGesture();
  this->cv_agent->stopGesture();
}

void Controller::startLongRecording() {
  auto dir = boost::filesystem::path(this->directory) / this->subject;
  int cv_index = 0, dvs_index = 0;
  boost::filesystem::path cv_path, dvs_path;

  // Ensure that directory exists
  boost::filesystem::create_directories(dir);

  // Find first non-existant total file
  do {
    cv_path = dir / ("total-" + std::to_string(cv_index) + ".mkv");
    cv_index += 1;
  } while (boost::filesystem::exists(cv_path));

  do {
    dvs_path = dir / ("total-" + std::to_string(dvs_index) + ".aedat");
    dvs_index += 1;
  } while (boost::filesystem::exists(dvs_path));

  try {
    this->dvs_agent->startLongRecording(dvs_path.string());
    this->cv_agent->startLongRecording(cv_path.string());
  } catch (std::runtime_error &e) {
    std::cout << e.what() << std::endl;
  }
}

void Controller::stopLongRecording() {
  this->dvs_agent->stopLongRecording();
  this->cv_agent->stopLongRecording();
}

void Controller::updateLabels() {
  if (this->num_gestures > 0) {
    auto label = std::to_string(this->current + 1) + " / " +
                 std::to_string(this->num_gestures);
    this->counter_label->SetLabel(label);

    auto name = std::get<0>(this->gestures[this->current]);

    if (this->currentFileExists()) {
      this->gesture_label->SetLabel(name + " (exists)");

      this->replay_button->Enable();
    } else {
      this->gesture_label->SetLabel(name);

      this->replay_button->Disable();
    }

    if (this->recording) {
      this->record_button->SetLabel("Stop Recording");
    } else {
      this->record_button->SetLabel("Start Recording");
    }

    if (this->long_recording) {
      this->long_record_button->SetLabel("Stop long recording");
    } else {
      this->long_record_button->SetLabel("Start long recording");
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

void Controller::playCurrentInstruction() {
  this->instructor->playInstructions(
      std::get<2>(this->gestures[this->current]));
}
}
}
