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
  if (!events || events->size() == 0) {
    this->events = nullptr;
    this->events_start = -1;
    this->events_end = -1;
  } else {
    this->events = events;
    this->events_start = 0;
    this->events_end = 0;
  }

  if (!frames || frames->size() == 0) {
    this->frames = nullptr;
    this->frame_index = -1;
  } else {
    this->frames = frames;
    this->frame_index = 0;
  }

  if (this->events && this->frames) {
    while ((*this->events)[this->events_end].timestamp <
           (*this->frames)[0].timestamp) {
      ++this->events_end;
    }
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

void DAVISWindow::setXRange(const float x_range) { this->x_range = x_range; }
void DAVISWindow::setYRange(const float y_range) { this->y_range = y_range; }

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
  const float aspect_ratio = (float)width / (float)height;

  if (this->frames) {
    const auto index = std::min(this->frame_index, this->frames->size() - 1);
    const auto &frame = (*this->frames)[index].frame;
    const float frame_as = frame.cols / (float)frame.rows;
    int x_offset = 0, y_offset = 0;
    int im_width = width, im_height = height;

    if (frame_as > aspect_ratio) {
      im_height = im_width / frame_as;
      y_offset = (height - im_height) / 2;
    } else {
      im_width = im_height * frame_as;
      x_offset = (width - im_width) / 2;
    }

    // Scale to client size
    wxImage im(frame.cols, frame.rows, frame.data, true);
    im = im.Scale(im_width, im_height);

    wxBitmap bm(im);
    dc.DrawBitmap(bm, x_offset, y_offset, false);
  }

  const auto X_RANGE = this->x_range;
  const auto Y_RANGE = this->y_range;
  const auto X_MAX = X_RANGE - 1;
  const auto Y_MAX = Y_RANGE - 1;

  if (this->events) {
    const float event_as = X_RANGE / (float)Y_RANGE;
    int x_offset = 0, y_offset = 0;
    auto width_ratio = width / X_RANGE;
    auto height_ratio = height / Y_RANGE;
    auto pixel_width = (wxCoord)std::max(width_ratio, 1.0f);
    auto pixel_height = (wxCoord)std::max(height_ratio, 1.0f);

    if (event_as > aspect_ratio) {
      height_ratio = width_ratio;
      pixel_height = pixel_width;
      y_offset = (height - Y_RANGE * height_ratio) / 2;
    } else {
      width_ratio = height_ratio;
      pixel_width = pixel_height;
      x_offset = (width - X_RANGE * width_ratio) / 2;
    }

    const auto end = std::min(this->events->size() - 1, this->events_end);
    const auto start = std::min(end, this->events_start);

    for (std::size_t i = start; i < end; ++i) {
      const auto &e = (*this->events)[i];

      const float x = (X_MAX - e.x) * width_ratio + x_offset;
      const float y = (Y_MAX - e.y) * height_ratio + y_offset;

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
