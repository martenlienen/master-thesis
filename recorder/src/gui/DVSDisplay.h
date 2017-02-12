#ifndef RECORDER_GUI_DVS_DISPLAY_H_
#define RECORDER_GUI_DVS_DISPLAY_H_

#include <mutex>
#include <vector>

#include <wx/timer.h>
#include <wx/window.h>

#include "edvs.h"

namespace recorder {

namespace gui {

class DVSDisplay : public wxWindow {
public:
  // Range of DVS128 event coordinates
  const int X_RANGE = 128;
  const int Y_RANGE = 128;

  DVSDisplay(wxWindow *parent, wxWindowID id,
             const wxPoint &pos = wxDefaultPosition,
             const wxSize &size = wxDefaultSize, long style = 0,
             const wxString &name = wxPanelNameStr);

  void pushEvents(std::vector<edvs_event_t> events);

protected:
  virtual wxSize DoGetBestSize() const;

private:
  wxTimer repaint_timer;
  std::mutex events_mutex;
  std::vector<std::vector<edvs_event_t>> events;
  uint64_t latest_t;

  void paintEvents();
};
}
}

#endif // RECORDER_GUI_DVS_DISPLAY_H_
