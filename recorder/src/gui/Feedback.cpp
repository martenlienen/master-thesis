#include <opencv2/imgproc.hpp>
#include <wx/dcclient.h>
#include <wx/timer.h>

#include "Feedback.h"

namespace recorder {

namespace gui {

Feedback::Feedback(capture::OpenCVCapture &cv_capture)
    : wxFrame(NULL, wxID_ANY, "Feedback"), cv_capture(cv_capture),
      repaint_timer(this) {
  this->Bind(wxEVT_PAINT, [this](wxPaintEvent &e) { this->onPaint(); });
  this->Bind(wxEVT_TIMER, [this](wxTimerEvent &e) { this->Refresh(); });

  // 30 FPS
  this->repaint_timer.Start(33.33333);
}

void Feedback::onPaint() {
  cv::Mat frame;
  {
    std::lock_guard<std::mutex> guard(this->cv_capture.last_frame_mutex);
    frame = this->cv_capture.last_frame.clone();
  }

  // This has to be instantiated in a PAINT event handler
  wxPaintDC dc(this);

  if (!frame.data) {
    return;
  }

  // Convert to the usual RGB format
  cv::cvtColor(frame, frame, cv::ColorConversionCodes::COLOR_BGR2RGB);

  wxImage im(frame.cols, frame.rows, frame.data, true);
  wxBitmap bm(im);
  dc.DrawBitmap(bm, 0, 0, false);
}
}
}
