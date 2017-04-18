#ifndef JAERREC_MAIN_FRAME_H_
#define JAERREC_MAIN_FRAME_H_

#include <cstdint>
#include <string>
#include <vector>

#include <wx/button.h>
#include <wx/frame.h>
#include <wx/mediactrl.h>
#include <wx/stattext.h>

namespace jaerrec {

class MainFrame : public wxFrame {
public:
  MainFrame();

  void nextGesture();

private:
  wxMediaCtrl *player;
  wxStaticText *header;
  wxButton *stop_button;

  std::string subject = "";
  std::string ip = "";
  std::uint16_t port = 8997;
  std::vector<std::string> gestures;
  std::string instruction_dir = "";

  int curr_gesture = -1;

  void startRecording();
  void stopRecording();

  void sendCommand(const std::string command) const;

  void updateView();
};
}

#endif // JAERREC_MAIN_FRAME_H_
