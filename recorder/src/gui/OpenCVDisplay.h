#ifndef RECORDER_GUI_OPENCV_DISPLAY_H_
#define RECORDER_GUI_OPENCV_DISPLAY_H_

#include <mutex>

#include <opencv2/imgproc.hpp>
#include <wx/timer.h>
#include <wx/window.h>

namespace recorder {

namespace gui {

class OpenCVDisplay : public wxWindow {
public:
  OpenCVDisplay(wxWindow *parent, wxWindowID id,
                const wxPoint &pos = wxDefaultPosition,
                const wxSize &size = wxDefaultSize, long style = 0,
                const wxString &name = wxPanelNameStr);

  void setFrame(cv::Mat frame);

protected:
  virtual wxSize DoGetBestSize() const;

private:
  wxTimer repaint_timer;
  mutable std::mutex frame_mutex;
  cv::Mat frame;

  void paintFrame();
};
}
}

#endif // RECORDER_GUI_OPENCV_DISPLAY_H_
