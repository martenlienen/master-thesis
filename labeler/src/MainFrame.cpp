#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <iomanip>
#include <sstream>

#include <boost/range/iterator_range.hpp>
#include <csv.h>
#include <opencv2/highgui/highgui.hpp>
#include <wx/menu.h>
#include <wx/msgdlg.h>
#include <wx/sizer.h>

// Somehow wx/dirdlg.h has to be included after wx/menu.h
#include <wx/dirdlg.h>

#include "MainFrame.h"

namespace labeler {

MainFrame::MainFrame() : wxFrame(NULL, wxID_ANY, "Label Stuff!") {
  this->davis = new DAVISWindow(this, wxID_ANY, wxDefaultPosition,
                                wxDefaultSize, 0, "davis");

  this->time_slider = new wxSlider(this, wxID_ANY, 0, 0, 1);
  this->time_label = new wxStaticText(this, wxID_ANY, "0:00 / 0:00");

  this->play_pause_button = new wxButton(this, wxID_ANY, "Play");
  auto reverse_button = new wxButton(this, wxID_ANY, "* -1");
  auto faster_button = new wxButton(this, wxID_ANY, "+10%");
  auto slower_button = new wxButton(this, wxID_ANY, "-10%");
  auto reset_speed_button = new wxButton(this, wxID_ANY, "=1");

  this->label_button = new wxButton(this, wxID_ANY, "New Label");
  this->label_input = new wxComboBox(this, wxID_ANY);

  this->label_delete_button = new wxButton(this, wxID_ANY, "Delete Label");
  this->current_label_label = new wxStaticText(this, wxID_ANY, "None");

  auto main_sizer = new wxBoxSizer(wxVERTICAL);
  auto slider_sizer = new wxBoxSizer(wxHORIZONTAL);
  auto player_buttons_sizer = new wxBoxSizer(wxHORIZONTAL);
  auto label_buttons_sizer = new wxBoxSizer(wxHORIZONTAL);
  auto current_label_sizer = new wxBoxSizer(wxHORIZONTAL);

  // Compose time slider line
  slider_sizer->Add(this->time_slider, wxSizerFlags().Expand().Proportion(1));
  slider_sizer->Add(this->time_label, wxSizerFlags().Expand());

  // Compose play/ff buttons etc.
  player_buttons_sizer->AddStretchSpacer();
  player_buttons_sizer->Add(this->play_pause_button, wxSizerFlags().Expand());
  player_buttons_sizer->Add(reverse_button, wxSizerFlags().Expand());
  player_buttons_sizer->Add(faster_button, wxSizerFlags().Expand());
  player_buttons_sizer->Add(slower_button, wxSizerFlags().Expand());
  player_buttons_sizer->Add(reset_speed_button, wxSizerFlags().Expand());
  player_buttons_sizer->AddStretchSpacer();

  // Compose labeling buttons
  label_buttons_sizer->AddStretchSpacer();
  label_buttons_sizer->Add(this->label_button, wxSizerFlags().Expand());
  label_buttons_sizer->Add(this->label_input, wxSizerFlags().Expand());
  label_buttons_sizer->AddStretchSpacer();

  // Compose current label info and actions
  current_label_sizer->AddStretchSpacer();
  current_label_sizer->Add(this->label_delete_button, wxSizerFlags().Expand());
  current_label_sizer->Add(this->current_label_label, wxSizerFlags().Expand());
  current_label_sizer->AddStretchSpacer();

  // Compose top-level components
  main_sizer->Add(this->davis, wxSizerFlags().Expand().Proportion(1));
  main_sizer->Add(slider_sizer, wxSizerFlags().Expand());
  main_sizer->Add(player_buttons_sizer, wxSizerFlags().Expand());
  main_sizer->Add(label_buttons_sizer, wxSizerFlags().Expand());
  main_sizer->Add(current_label_sizer, wxSizerFlags().Expand());

  this->SetSizerAndFit(main_sizer);

  // Set up the menu bar
  auto menu_bar = new wxMenuBar();
  auto file_menu = new wxMenu();
  file_menu->Append(wxID_OPEN, "&Open");
  file_menu->Append(wxID_SAVE, "&Save");
  menu_bar->Append(file_menu, "&File");
  this->SetMenuBar(menu_bar);

  this->Bind(wxEVT_MENU,
             [this](const wxCommandEvent &e) {
               auto dlg =
                   new wxDirDialog(this, "Select recording to label", "",
                                   wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

               if (dlg->ShowModal() == wxID_OK) {
                 boost::filesystem::path dir(dlg->GetPath());
                 auto events_path = dir / "events.csv";
                 auto frames_path = dir / "frames";

                 if (!(boost::filesystem::is_regular_file(events_path))) {
                   auto msg_dlg = new wxMessageDialog(
                       this, "This directory does not contain a recording",
                       "Message", wxOK);
                   msg_dlg->ShowModal();
                   return;
                 }

                 this->openRecordings(dir);
               }
             },
             wxID_OPEN, wxID_OPEN);
  this->Bind(
      wxEVT_MENU,
      [this](const wxCommandEvent &e) {
        if (this->dir.empty()) {
          return;
        }

        auto out_file = this->dir / "labels.csv";

        if (boost::filesystem::exists(out_file)) {
          if (wxMessageBox("Do you want to overwrite the existing labels?",
                           "Message", wxYES_NO) != wxYES) {
            return;
          }
        }

        std::ofstream out(out_file.string());

        if (!out.is_open()) {
          wxMessageBox("Could not open labels.csv", "Message", wxOK);
          return;
        }

        out << "start,end,name\n";

        for (auto &label : this->labels) {
          out << label.start << "," << label.end << "," << label.label << "\n";
        }
      },
      wxID_SAVE, wxID_SAVE);

  this->time_slider->Bind(wxEVT_SLIDER, [this](const wxCommandEvent &e) {
    this->setTime(this->min_time + e.GetInt());
  });

  this->play_pause_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->isPlaying()) {
      this->pause();
    } else {
      this->play();
    }

    this->updateLabels();
  });

  reverse_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    this->setSpeed(-1.0 * this->speed);
  });
  faster_button->Bind(wxEVT_BUTTON,
                      [this](const wxCommandEvent &e) { this->faster(); });
  slower_button->Bind(wxEVT_BUTTON,
                      [this](const wxCommandEvent &e) { this->slower(); });
  reset_speed_button->Bind(
      wxEVT_BUTTON, [this](const wxCommandEvent &e) { this->setSpeed(NORMAL_SPEED); });

  this->label_button->Bind(wxEVT_BUTTON, [this](const wxCommandEvent &e) {
    if (this->labeling_state == NONE) {
      this->labeling_state = FINDING_START;

      this->setSpeed(-SLOW_MOTION_SPEED);
    } else if (this->labeling_state == FINDING_START) {
      this->labeling_state = FINDING_END;
      this->candidate_label.start = this->time;

      this->setSpeed(SLOW_MOTION_SPEED);
    } else {
      this->labeling_state = NONE;
      this->candidate_label.end = this->time;
      this->candidate_label.label = this->label_input->GetValue();
      this->labels.push_back(this->candidate_label);

      // Keep labels sorted by start time
      std::sort(std::begin(labels), std::end(labels),
                [](const StreamLabel &a, const StreamLabel &b) {
                  return a.start < b.start;
                });

      // Select next label
      auto selection = this->label_input->GetSelection();
      if (selection != wxNOT_FOUND) {
        this->label_input->SetSelection(selection + 1);
      }

      this->setSpeed(NORMAL_SPEED);
      this->play();
    }

    this->updateLabels();
  });

  this->label_delete_button->Bind(
      wxEVT_BUTTON, [this](const wxCommandEvent &e) {
        for (auto it = std::begin(this->labels); it != std::end(this->labels);
             ++it) {
          if ((*it).start <= this->time && this->time <= (*it).end) {
            this->labels.erase(it);
            this->updateLabels();
            return;
          }
        }
      });
}

void MainFrame::openRecordings(const boost::filesystem::path dir) {
  this->dir = dir;
  const auto events_path = dir / "events.csv";
  const auto frames_path = dir / "frames";
  const auto labels_path = dir / "labels.csv";
  const auto log_path = dir / "recording-log";

  auto events_future = std::async([events_path]() {
    try {
      std::vector<StreamEvent> events;
      io::CSVReader<4> event_reader(events_path.string());
      event_reader.read_header(io::ignore_extra_column, "timestamp", "x", "y",
                               "polarity");
      std::uint64_t timestamp;
      std::uint16_t x, y;
      std::uint8_t polarity;
      while (event_reader.read_row(timestamp, x, y, polarity)) {
        events.push_back({timestamp, x, y, polarity});
      }
      return events;
    } catch (io::error::base &e) {
      // Something is wrong with the csv file. Ignore.
      std::cout << e.what() << std::endl;
      return std::vector<StreamEvent>();
    }
  });

  auto frames_future = std::async([frames_path]() {
    std::vector<StreamFrame> frames;

    // In a DVS recording there are no frames
    if (!boost::filesystem::exists(frames_path)) {
      return frames;
    }

    boost::filesystem::directory_iterator it(frames_path);
    for (auto &dir_entry : boost::make_iterator_range(it, {})) {
      auto frame_path = dir_entry.path();
      std::uint64_t timestamp(std::stoll(frame_path.stem().string()));

      // We load them as color images because, we need them in BGR format to
      // convert them to wxImages for displaying
      auto frame = cv::imread(frame_path.string(), CV_LOAD_IMAGE_COLOR);

      frames.push_back({timestamp, frame});
    }
    return frames;
  });

  auto labels_future = std::async([labels_path]() {
    std::vector<StreamLabel> labels;

    if (!boost::filesystem::exists(labels_path)) {
      return labels;
    }

    try {
      io::CSVReader<3> label_reader(labels_path.string());
      label_reader.read_header(io::ignore_extra_column, "start", "end", "name");
      std::uint64_t start, end;
      std::string name;
      while (label_reader.read_row(start, end, name)) {
        labels.push_back({start, end, name});
      }
    } catch (io::error::base &e) {
      // Something is wrong with the csv file. Ignore.
      std::cout << e.what() << std::endl;
    }

    std::sort(std::begin(labels), std::end(labels),
              [](const StreamLabel &a, const StreamLabel &b) {
                return a.start < b.start;
              });

    return labels;
  });

  if (boost::filesystem::exists(log_path)) {
    this->label_input->Clear();
    std::ifstream in(log_path.string());

    if (in.is_open()) {
      std::string name;

      while (std::getline(in, name)) {
        this->label_input->Append(name);
      }

      this->label_input->SetSelection(0);
    }
  }

  this->events = events_future.get();
  this->frames = frames_future.get();
  this->labels = labels_future.get();

  std::sort(std::begin(this->frames), std::end(this->frames),
            [](const StreamFrame &a, const StreamFrame &b) {
              return a.timestamp < b.timestamp;
            });

  this->davis->reset(&events, &frames);
  const auto max_x = std::max_element(
      std::begin(events), std::end(events),
      [](const StreamEvent a, const StreamEvent b) { return a.x < b.x; });
  if (max_x != std::end(events) && max_x->x < 128) {
    // DVS
    this->davis->setXRange(128.0);
    this->davis->setYRange(128.0);
  } else {
    // DAVIS
    this->davis->setXRange(240.0);
    this->davis->setYRange(180.0);
  }

  this->min_time = this->events[0].timestamp;
  this->max_time = this->events.back().timestamp;

  if (this->frames.size() > 0) {
    this->min_time = std::min(this->min_time, this->frames[0].timestamp);
    this->max_time = std::max(this->max_time, this->frames.back().timestamp);
  }

  this->time = this->min_time;

  // Shift the range to start at 0 because for long recordings the timestamps
  // might actually exceed INT_MAX
  this->time_slider->SetRange(0, max_time - min_time);
}

void MainFrame::setTime(std::uint64_t time) {
  time = std::min(this->max_time, std::max(this->min_time, time));

  this->time = time;
  this->time_slider->SetValue(time - this->min_time);
  this->davis->setTime(time);

  // Update the time label
  const auto duration = this->max_time - this->min_time;
  const auto passed = this->time - this->min_time;
  const auto label =
      this->formatTime(passed) + " / " + this->formatTime(duration);
  this->time_label->SetLabel(label);

  this->updateLabels();
}

void MainFrame::advanceTime(std::int64_t delta) {
  this->setTime(this->time + delta);
}

void MainFrame::play() {
  // Play at 30 FPS
  this->play_timer.Start(33);
}
void MainFrame::pause() { this->play_timer.Stop(); }
bool MainFrame::isPlaying() { return this->play_timer.IsRunning(); }
void MainFrame::setSpeed(float speed) {
  this->speed = speed;
  this->play_timer.setSpeed(speed);

  // Scale the number of events with the speed so that faster play speeds show
  // more events per frame and vice versa
  this->davis->setEventAccumulationTime(33000 * std::abs(speed));
}
void MainFrame::faster() {
  this->setSpeed(this->speed + std::copysign(0.1, this->speed));
};
void MainFrame::slower() {
  this->setSpeed(this->speed - std::copysign(0.1, this->speed));
};

std::experimental::optional<StreamLabel> MainFrame::currentLabel() const {
  for (auto &label : this->labels) {
    if (label.start <= this->time && this->time <= label.end) {
      return {label};
    }
  }

  return {};
}

std::string MainFrame::formatTime(const std::uint64_t time) const {
  const std::uint64_t MICROSEC_PER_S = 1000000;
  const auto seconds = time / MICROSEC_PER_S;
  const auto time_m = seconds / 60;
  const auto time_s = seconds % 60;

  std::stringstream out;
  out << time_m << ":" << std::setfill('0') << std::setw(2) << time_s;

  return out.str();
}

void MainFrame::updateLabels() {
  if (this->isPlaying()) {
    this->play_pause_button->SetLabel("Pause");
  } else {
    this->play_pause_button->SetLabel("Play");
  }

  switch (this->labeling_state) {
  case NONE:
    this->label_button->SetLabel("New Label");
    break;
  case FINDING_START:
    this->label_button->SetLabel("Mark Start");
    break;
  case FINDING_END:
    this->label_button->SetLabel("Mark End");
    break;
  }

  auto label = this->currentLabel();
  if (label) {
    this->label_delete_button->Enable();
    this->current_label_label->SetLabel(
        label->label + " (" + this->formatTime(label->start - this->min_time) +
        " - " + this->formatTime(label->end - this->min_time) + ")");
  } else {
    this->label_delete_button->Disable();
    this->current_label_label->SetLabel("None");
  }
}
}
