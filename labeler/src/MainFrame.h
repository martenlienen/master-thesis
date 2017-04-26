#ifndef LABELER_MAIN_FRAME_H_
#define LABELER_MAIN_FRAME_H_

#include <experimental/optional>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <wx/button.h>
#include <wx/combobox.h>
#include <wx/frame.h>
#include <wx/slider.h>
#include <wx/stattext.h>

#include "DAVISWindow.h"
#include "PlayTimer.h"
#include "Stream.h"

namespace labeler {

enum LabelingState { NONE, FINDING_START, FINDING_END };

class MainFrame : public wxFrame {
public:
  const float NORMAL_SPEED = 1.5;
  const float SLOW_MOTION_SPEED = 0.5;

  MainFrame();

private:
  DAVISWindow *davis;
  wxSlider *time_slider;
  wxStaticText *time_label;
  wxButton *play_pause_button;
  wxButton *label_button;
  wxButton *label_delete_button;
  wxComboBox *label_input;
  wxStaticText *current_label_label;

  boost::filesystem::path dir;

  std::uint64_t time = 0;
  std::uint64_t min_time = 0;
  std::uint64_t max_time = 0;
  float speed = 1.0;
  PlayTimer play_timer{
      [this](std::int64_t delta) { this->advanceTime(delta); }};

  LabelingState labeling_state = NONE;
  StreamLabel candidate_label;

  std::vector<StreamEvent> events;
  std::vector<StreamFrame> frames;
  std::vector<StreamLabel> labels;

  void openRecordings(boost::filesystem::path dir);

  void setTime(std::uint64_t delta);
  void advanceTime(std::int64_t delta);

  void play();
  void pause();
  bool isPlaying();
  void setSpeed(float speed);
  void faster();
  void slower();

  // Returns the label for the current time if there is one
  std::experimental::optional<StreamLabel> currentLabel() const;

  std::string formatTime(const std::uint64_t time) const;

  void updateLabels();
};
}

#endif // LABELER_MAIN_FRAME_H_
