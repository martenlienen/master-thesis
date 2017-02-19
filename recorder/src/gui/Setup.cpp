#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
#include <csv.h>
#include <wx/button.h>
#include <wx/choice.h>
#include <wx/filepicker.h>
#include <wx/msgdlg.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>

#include "../agents/DVSAgent.h"
#include "../agents/OpenCVAgent.h"
#include "Controller.h"
#include "Setup.h"

namespace recorder {

namespace gui {

Setup::Setup() : wxFrame(NULL, wxID_ANY, "Setup") {
  auto num_cameras = capture::OpenCVCapture::getNumCameras();
  this->readSettings(num_cameras);

  auto camera_choices = wxArrayString();
  for (auto i = 0; i < num_cameras; i++) {
    camera_choices.Add(std::to_string(i));
  }

  auto rotate_choices = wxArrayString();
  rotate_choices.Add("0");
  rotate_choices.Add("180");

  // Create controls
  auto name_label = new wxStaticText(this, wxID_ANY, "Subject Name");
  auto name_text = new wxTextCtrl(this, wxID_ANY, this->subject);
  auto camera_label = new wxStaticText(this, wxID_ANY, "Camera");
  auto camera_choice = new wxChoice(this, wxID_ANY, wxDefaultPosition,
                                    wxDefaultSize, camera_choices);
  camera_choice->SetSelection(this->camera_id);
  auto rotate_label = new wxStaticText(this, wxID_ANY, "Camera Rotation");
  auto rotate_choice = new wxChoice(this, wxID_ANY, wxDefaultPosition,
                                    wxDefaultSize, rotate_choices);
  if (this->rotate_degrees == 0) {
    rotate_choice->SetSelection(0);
  } else {
    rotate_choice->SetSelection(1);
  }
  auto dvs_label = new wxStaticText(this, wxID_ANY, "DVS Device");
  auto dvs_picker = new wxFilePickerCtrl(
      this, wxID_ANY, this->dvs_device, "Which DVS device?",
      wxFileSelectorDefaultWildcardStr, wxDefaultPosition, wxDefaultSize,
      wxFLP_USE_TEXTCTRL | wxFLP_OPEN);
  auto dir_label = new wxStaticText(this, wxID_ANY, "Storage Directory");
  auto dir_picker = new wxDirPickerCtrl(
      this, wxID_ANY, this->directory, "Where to save data?", wxDefaultPosition,
      wxDefaultSize, wxDIRP_USE_TEXTCTRL);
  // The constructor ignores the initial value
  dir_picker->SetPath(this->directory);
  auto gestures_label = new wxStaticText(this, wxID_ANY, "Gestures File");
  auto gestures_picker =
      new wxFilePickerCtrl(this, wxID_ANY, this->gestures, "Gestures File?",
                           "Gesture CSV files (*.csv)|*.csv", wxDefaultPosition,
                           wxDefaultSize, wxFLP_USE_TEXTCTRL | wxFLP_OPEN);
  // The constructor ignores the initial value
  gestures_picker->SetPath(this->gestures);
  auto camera_preview_btn = new wxButton(this, wxID_ANY, "Camera Preview");
  auto dvs_preview_btn = new wxButton(this, wxID_ANY, "DVS Preview");
  auto go_btn = new wxButton(this, wxID_ANY, "Start Recording");
  camera_choice->SetSelection(0);

  // Arrange controls
  auto top_sizer = new wxBoxSizer(wxVERTICAL);
  auto form_sizer = new wxGridSizer(2, 10, 10);
  form_sizer->Add(name_label, wxSizerFlags().Center());
  form_sizer->Add(name_text, wxSizerFlags().Center().Expand());
  form_sizer->Add(camera_label, wxSizerFlags().Center());
  form_sizer->Add(camera_choice, wxSizerFlags().Center().Expand());
  form_sizer->Add(rotate_label, wxSizerFlags().Center());
  form_sizer->Add(rotate_choice, wxSizerFlags().Center().Expand());
  form_sizer->Add(dvs_label, wxSizerFlags().Center());
  form_sizer->Add(dvs_picker, wxSizerFlags().Center().Expand());
  form_sizer->Add(dir_label, wxSizerFlags().Center());
  form_sizer->Add(dir_picker, wxSizerFlags().Center().Expand());
  form_sizer->Add(gestures_label, wxSizerFlags().Center());
  form_sizer->Add(gestures_picker, wxSizerFlags().Center().Expand());
  top_sizer->Add(form_sizer, 1, wxEXPAND | wxALL, 10);
  const auto btn_flags =
      wxSizerFlags().Proportion(0).Expand().Border(wxALL, 10);
  top_sizer->Add(camera_preview_btn, btn_flags);
  top_sizer->Add(dvs_preview_btn, btn_flags);
  top_sizer->Add(go_btn, btn_flags);

  this->SetSizerAndFit(top_sizer);

  // Set up event handling
  name_text->Bind(wxEVT_TEXT, [this](const wxCommandEvent &e) {
    this->subject = e.GetString();
  });
  camera_choice->Bind(wxEVT_CHOICE, [this](const wxCommandEvent &e) {
    this->camera_id = e.GetInt();
  });
  rotate_choice->Bind(wxEVT_CHOICE, [this](const wxCommandEvent &e) {
    if (e.GetInt() == 0) {
      this->rotate_degrees = 0;
    } else {
      this->rotate_degrees = 180;
    }
  });
  dvs_picker->Bind(wxEVT_FILEPICKER_CHANGED,
                   [this](const wxFileDirPickerEvent &e) {
                     this->dvs_device = e.GetPath();
                   });
  dir_picker->Bind(
      wxEVT_DIRPICKER_CHANGED,
      [this](const wxFileDirPickerEvent &e) { this->directory = e.GetPath(); });
  gestures_picker->Bind(
      wxEVT_FILEPICKER_CHANGED,
      [this](const wxFileDirPickerEvent &e) { this->gestures = e.GetPath(); });
  go_btn->Bind(wxEVT_BUTTON,
               [this](const wxCommandEvent &e) { this->startController(); });
  camera_preview_btn->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {});
  dvs_preview_btn->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {});
}

void Setup::startController() {
  this->writeSettings();

  try {
    auto gestures = this->parseGestures();

    std::unique_ptr<agents::DVSAgent> dvs_agent(new agents::DVSAgent());
    std::unique_ptr<agents::OpenCVAgent> cv_agent(
        new agents::OpenCVAgent(this->rotate_degrees));

    dvs_agent->start(this->dvs_device);
    cv_agent->start(this->camera_id);

    auto controller = new Controller(this->subject, this->directory, gestures,
                                     std::move(dvs_agent), std::move(cv_agent));
    controller->Show();
    this->Close();
  } catch (io::error::base &e) {
    auto msg = std::string("Could not open gestures file:\n") + e.what();
    auto dlg = new wxMessageDialog(this, msg);
    dlg->ShowModal();
    return;
  } catch (std::runtime_error &e) {
    auto msg = std::string("Could not open DVS:\n") + e.what();
    auto dlg = new wxMessageDialog(this, msg);
    dlg->ShowModal();
    return;
  }
}

std::vector<std::tuple<std::string, std::string, std::string>>
Setup::parseGestures() {
  boost::filesystem::path path(this->gestures);
  boost::filesystem::path dir = path.parent_path();
  io::CSVReader<3> reader(path.string());
  reader.set_header("Name", "id", "file");

  std::vector<std::tuple<std::string, std::string, std::string>> lines;
  std::string name, id, file;
  while (reader.read_row(name, id, file)) {
    boost::filesystem::path file_path(dir);
    file_path += file;
    lines.push_back(std::make_tuple(name, id, file_path.string()));
  }

  return lines;
}

void Setup::readSettings(uint32_t num_cameras) {
  boost::filesystem::path path(SETTINGS_PATH);
  if (!boost::filesystem::exists(path)) {
    return;
  }

  std::ifstream stream(path.string());

  if (!stream.is_open()) {
    return;
  }

  // This has to be the same order as in #writeSettings
  std::getline(stream, this->subject);
  std::string camera_id_buf;
  std::getline(stream, camera_id_buf);
  auto camera_id = std::stoi(camera_id_buf);
  if (camera_id <= num_cameras) {
    this->camera_id = camera_id;
  }
  std::string rotate_buf;
  std::getline(stream, rotate_buf);
  auto rotate_deg = std::stoi(rotate_buf);
  if (rotate_deg == 0 || rotate_deg == 180) {
    this->rotate_degrees = rotate_deg;
  }
  std::getline(stream, this->dvs_device);
  std::getline(stream, this->directory);
  std::getline(stream, this->gestures);
}

void Setup::writeSettings() {
  std::ofstream stream(SETTINGS_PATH);

  if (!stream.is_open()) {
    return;
  }

  stream << this->subject << std::endl;
  stream << this->camera_id << std::endl;
  stream << this->rotate_degrees << std::endl;
  stream << this->dvs_device << std::endl;
  stream << this->directory << std::endl;
  stream << this->gestures << std::endl;
}
}
}
