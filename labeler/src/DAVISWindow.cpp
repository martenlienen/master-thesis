#include <algorithm>

#include <wx/dcclient.h>

#include "DAVISWindow.h"

namespace labeler {

DAVISWindow::DAVISWindow(wxWindow *parent, wxWindowID id, const wxPoint &pos,
                         const wxSize &size, long style, const wxString &name)
    : wxWindow(parent, id, pos, size, style, name), events(nullptr),
      frames(nullptr), frame_index(-1), events_start(-1), events_end(-1) {
  this->Bind(wxEVT_PAINT, [this](wxPaintEvent &e) { this->paintFrame(); });
}

void DAVISWindow::reset(const std::vector<StreamEvent> *events,
                        const std::vector<StreamFrame> *frames) {
  if (!events || !frames || events->size() == 0 || frames->size() == 0) {
    this->events = nullptr;
    this->frames = nullptr;
    this->frame_index = -1;
    this->events_start = -1;
    this->events_end = -1;
    return;
  }

  this->events = events;
  this->frames = frames;
  this->frame_index = 0;
  this->events_start = 0;
  this->events_end = 0;

  while ((*this->events)[this->events_end].timestamp <
         (*this->frames)[0].timestamp) {
    ++this->events_end;
  }

  this->InvalidateBestSize();
  this->Refresh();
}

void DAVISWindow::setTime(const std::uint64_t time) {
  if (this->frames) {
    // Find frame at time or directly before time
    this->frame_index =
        this->binarySearch(*this->frames, time, 0, this->frames->size() - 1);
  }

  if (this->events) {
    // Find event at time or directly before time
    this->events_end =
        this->binarySearch(*this->events, time, 0, this->events->size() - 1);

    // Find first event that is at most this->event_accumulation_time before the
    // current time
    if (time <= this->event_accumulation_time) {
      this->events_start = 0;
    } else {
      this->events_start = this->binarySearch(
          *this->events, time - this->event_accumulation_time, 0,
          this->events_end);
    }
  }

  this->Refresh();
}

void DAVISWindow::setEventAccumulationTime(const std::uint64_t time) {
  this->event_accumulation_time = time;
}

wxSize DAVISWindow::DoGetBestSize() const {
  if (this->frames && this->frames->size() > 0) {
    return wxSize((*this->frames)[0].frame.cols, (*this->frames)[0].frame.rows);
  } else {
    return wxSize(240, 180);
  }
}

void DAVISWindow::paintFrame() {
  // This has to be instantiated in any PAINT event handler
  wxPaintDC dc(this);

  int width, height;
  this->GetClientSize(&width, &height);

  if (this->frames) {
    const auto index = std::min(this->frame_index, this->frames->size() - 1);
    const auto &frame = (*this->frames)[index].frame;

    // Scale to client size
    wxImage im(frame.cols, frame.rows, frame.data, true);
    im = im.Scale(width, height);

    wxBitmap bm(im);
    dc.DrawBitmap(bm, 0, 0, false);
  }

  if (this->events) {
    const auto width_ratio = width / X_RANGE;
    const auto height_ratio = height / Y_RANGE;
    const auto pixel_width = (wxCoord)std::max(width_ratio, 1.0f);
    const auto pixel_height = (wxCoord)std::max(height_ratio, 1.0f);

    const auto end = std::min(this->events->size() - 1, this->events_end);
    const auto start = std::min(end, this->events_start);

    for (std::size_t i = start; i < end; ++i) {
      const auto &e = (*this->events)[i];

      const float x = (X_RANGE - e.x) * width_ratio;
      const float y = (Y_RANGE - e.y) * height_ratio;

      if (e.parity) {
        dc.SetPen(*wxGREEN_PEN);
        dc.SetBrush(*wxGREEN_BRUSH);
      } else {
        dc.SetPen(*wxRED_PEN);
        dc.SetBrush(*wxRED_BRUSH);
      }

      dc.DrawRectangle((wxCoord)x, (wxCoord)y, pixel_width, pixel_height);
    }
  }
}

template <typename T>
std::size_t DAVISWindow::binarySearch(const std::vector<T> &data,
                                      std::uint64_t time, std::size_t min,
                                      std::size_t max) const {
  while (max > min) {
    const std::size_t index = (min + max) / 2;
    const auto &element = data[index];

    if (element.timestamp == time) {
      min = index;
      break;
    } else if (element.timestamp < time) {
      min = index + 1;
    } else {
      max = index - 1;
    }
  }

  return min;
}
}
