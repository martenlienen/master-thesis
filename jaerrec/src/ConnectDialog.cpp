#include <fstream>

#include <boost/filesystem.hpp>
#include <wx/button.h>
#include <wx/filepicker.h>
#include <wx/msgdlg.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>

#include "ConnectDialog.h"
#include "csv.h"

namespace jaerrec {

ConnectDialog::ConnectDialog(wxWindow *parent)
    : wxDialog(parent, wxID_ANY, "Start Recording", wxDefaultPosition,
               wxDefaultSize) {
  this->readCacheFile();

  auto subject_label = new wxStaticText(this, wxID_ANY, "Subject");
  auto subject_input = new wxTextCtrl(this, wxID_ANY, this->subject);
  auto ip_label = new wxStaticText(this, wxID_ANY, "jAER IP");
  auto ip_input = new wxTextCtrl(this, wxID_ANY, this->ip);
  auto port_label = new wxStaticText(this, wxID_ANY, "jAER Port");
  auto port_input = new wxTextCtrl(this, wxID_ANY, std::to_string(this->port));
  auto gesture_label = new wxStaticText(this, wxID_ANY, "Gesture File");
  auto gesture_input =
      new wxFilePickerCtrl(this, wxID_ANY, this->path, "Gesture File",
                           wxFileSelectorDefaultWildcardStr, wxDefaultPosition,
                           wxDefaultSize, wxFLP_USE_TEXTCTRL);
  // The constructor does not update the text field
  gesture_input->SetPath(this->path);
  auto dir_label = new wxStaticText(this, wxID_ANY, "Instruction Videos");
  auto dir_input = new wxDirPickerCtrl(this, wxID_ANY, this->instruction_dir,
                                       "", wxDefaultPosition, wxDefaultSize,
                                       wxDIRP_USE_TEXTCTRL);
  // The constructor does not update the text field
  dir_input->SetPath(this->instruction_dir);

  auto ok_button = new wxButton(this, wxID_OK, "OK");
  auto cancel_button = new wxButton(this, wxID_CANCEL, "Cancel");

  auto main_sizer = new wxBoxSizer(wxVERTICAL);
  auto form_sizer = new wxGridSizer(2, 10, 10);
  auto button_sizer = new wxBoxSizer(wxHORIZONTAL);

  auto expand_flags = wxSizerFlags().Expand();
  auto center_flags = expand_flags.Center();
  form_sizer->Add(subject_label, center_flags);
  form_sizer->Add(subject_input, center_flags);
  form_sizer->Add(ip_label, center_flags);
  form_sizer->Add(ip_input, center_flags);
  form_sizer->Add(port_label, center_flags);
  form_sizer->Add(port_input, center_flags);
  form_sizer->Add(gesture_label, center_flags);
  form_sizer->Add(gesture_input, center_flags);
  form_sizer->Add(dir_label, center_flags);
  form_sizer->Add(dir_input, center_flags);
  button_sizer->Add(ok_button);
  button_sizer->Add(cancel_button);
  main_sizer->Add(form_sizer, expand_flags);
  main_sizer->Add(button_sizer, expand_flags);

  this->SetSizerAndFit(main_sizer);

  subject_input->Bind(
      wxEVT_TEXT, [this](wxCommandEvent &e) { this->subject = e.GetString(); });
  ip_input->Bind(wxEVT_TEXT,
                 [this](wxCommandEvent &e) { this->ip = e.GetString(); });
  port_input->Bind(wxEVT_TEXT, [this](wxCommandEvent &e) {
    std::string buf = e.GetString().ToStdString();

    this->port = std::stoi(buf);
  });
  gesture_input->Bind(
      wxEVT_FILEPICKER_CHANGED,
      [this](wxFileDirPickerEvent &e) { this->path = e.GetPath(); });
  dir_input->Bind(wxEVT_DIRPICKER_CHANGED, [this](wxFileDirPickerEvent &e) {
    this->instruction_dir = e.GetPath();
  });

  ok_button->Bind(wxEVT_BUTTON, [this](wxCommandEvent &e) {
    // Try reading the gestures files
    try {
      this->gestures.clear();
      io::CSVReader<1> reader(this->path);
      reader.set_header("Name");
      std::string buf;
      while (reader.read_row(buf)) {
        this->gestures.push_back(buf);
      }

      this->writeCacheFile();

      // Bubble event up to close dialog
      e.Skip();
    } catch (io::error::base &e) {
      wxMessageBox(std::string("Could not read gesture file\n") + e.what());
    }
  });
}

void ConnectDialog::readCacheFile() {
  boost::filesystem::path path(CACHE_PATH);
  if (!boost::filesystem::exists(path)) {
    return;
  }

  std::ifstream stream(path.string());
  if (!stream.is_open()) {
    return;
  }

  std::string port_buf;
  std::getline(stream, this->subject);
  std::getline(stream, this->ip);
  std::getline(stream, port_buf);
  std::getline(stream, this->path);
  std::getline(stream, this->instruction_dir);

  this->port = std::stoi(port_buf);
}

void ConnectDialog::writeCacheFile() {
  std::ofstream stream(CACHE_PATH);

  stream << this->subject << std::endl;
  stream << this->ip << std::endl;
  stream << std::to_string(this->port) << std::endl;
  stream << this->path << std::endl;
  stream << this->instruction_dir << std::endl;
}
}
