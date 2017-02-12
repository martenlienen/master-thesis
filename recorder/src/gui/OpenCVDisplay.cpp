#include <opencv2/imgproc.hpp>
#include <wx/dcclient.h>

#include "OpenCVDisplay.h"

namespace recorder {

namespace gui {

OpenCVDisplay::OpenCVDisplay(wxWindow *parent, wxWindowID id,
                             const wxPoint &pos, const wxSize &size, long style,
                             const wxString &name)
    : wxWindow(parent, id, pos, size, style, name) {
  this->Bind(wxEVT_PAINT, [this](wxPaintEvent &e) { this->paintFrame(); });
  this->repaint_timer.Bind(wxEVT_TIMER,
                           [this](wxTimerEvent &e) { this->Refresh(); });

  // 30 FPS
  this->repaint_timer.Start(33.33333);
}

void OpenCVDisplay::setFrame(cv::Mat frame) {
  std::lock_guard<std::mutex> guard(this->frame_mutex);

  // Clone the frame, because we are manipulating it in-place
  this->frame = frame.clone();

  // Convert to the usual RGB format
  cv::cvtColor(this->frame, this->frame,
               cv::ColorConversionCodes::COLOR_BGR2RGB);

  this->InvalidateBestSize();
}

wxSize OpenCVDisplay::DoGetBestSize() const {
  std::lock_guard<std::mutex> guard(this->frame_mutex);

  if (this->frame.data) {
    return wxSize(this->frame.cols, this->frame.rows);
  } else {
    return wxSize(240, 180);
  }
}

void OpenCVDisplay::paintFrame() {
  // This has to be instantiated in any PAINT event handler
  wxPaintDC dc(this);

  if (!frame.data) {
    return;
  }

  int width, height;
  this->GetClientSize(&width, &height);

  std::lock_guard<std::mutex> guard(this->frame_mutex);

  // Scale to client size
  wxImage im(frame.cols, frame.rows, frame.data, true);
  im.Rescale(width, height);

  wxBitmap bm(im);
  dc.DrawBitmap(bm, 0, 0, false);
}
}
}
