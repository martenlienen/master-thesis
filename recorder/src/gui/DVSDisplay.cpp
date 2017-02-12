#include <algorithm>
#include <iostream>

#include <wx/dcclient.h>

#include "DVSDisplay.h"

namespace recorder {

namespace gui {

DVSDisplay::DVSDisplay(wxWindow *parent, wxWindowID id, const wxPoint &pos,
                       const wxSize &size, long style, const wxString &name)
    : wxWindow(parent, id, pos, size, style, name),
      events(X_RANGE, std::vector<edvs_event_t>(Y_RANGE)) {
  this->Bind(wxEVT_PAINT, [this](wxPaintEvent &e) { this->paintEvents(); });
  this->repaint_timer.Bind(wxEVT_TIMER,
                           [this](wxTimerEvent &e) { this->Refresh(); });

  // 30 FPS
  this->repaint_timer.Start(33.33333);
}

void DVSDisplay::pushEvents(std::vector<edvs_event_t> events) {
  std::lock_guard<std::mutex> guard(this->events_mutex);

  for (auto &e : events) {
    this->events[e.x][e.y] = e;
  }

  if (events.size() > 0) {
    this->latest_t = events.back().t;
  }
}

wxSize DVSDisplay::DoGetBestSize() const { return wxSize(X_RANGE, Y_RANGE); }

void DVSDisplay::paintEvents() {
  // This has to be instantiated in any PAINT event handler
  wxPaintDC dc(this);

  std::lock_guard<std::mutex> guard(this->events_mutex);

  int width, height;
  this->GetClientSize(&width, &height);

  const int pixel_width = std::max(1, width / X_RANGE);
  const int pixel_height = std::max(1, height / Y_RANGE);

  // Paint events
  const auto TTL = 33000; // microseconds = 0.033 seconds
  const auto latest = this->latest_t;
  for (int x = 0; x < X_RANGE; x++) {
    for (int y = 0; y < Y_RANGE; y++) {
      const auto e = this->events[x][y];

      // Discard old events
      if (latest - e.t > TTL) {
        continue;
      }

      int origin_x = (X_RANGE - x) * width / X_RANGE;
      int origin_y = (Y_RANGE - y) * height / Y_RANGE;

      if (e.parity) {
        dc.SetPen(*wxBLACK_PEN);
        dc.SetBrush(*wxBLACK_BRUSH);
      } else {
        dc.SetPen(*wxWHITE_PEN);
        dc.SetBrush(*wxWHITE_BRUSH);
      }

      dc.DrawRectangle(origin_x, origin_y, pixel_width, pixel_height);
    }
  }
}
}
}
