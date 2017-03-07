#ifndef LABELER_DAVIS_WINDOW_H_
#define LABELER_DAVIS_WINDOW_H_

#include <cstdint>
#include <vector>

#include <wx/window.h>

#include "Stream.h"

namespace labeler {

class DAVISWindow : public wxWindow {
public:
  const float X_RANGE = 240.0;
  const float Y_RANGE = 180.0;

  DAVISWindow(wxWindow *parent, wxWindowID id, const wxPoint &pos,
              const wxSize &size, long style, const wxString &name);

  void reset(const std::vector<StreamEvent> *events,
             const std::vector<StreamFrame> *frames);

  void setTime(const std::uint64_t time);
  void setEventAccumulationTime(const std::uint64_t time);

protected:
  virtual wxSize DoGetBestSize() const;

private:
  const std::vector<StreamEvent> *events;
  const std::vector<StreamFrame> *frames;
  std::size_t frame_index;
  std::size_t events_start;
  std::size_t events_end;

  // By default accumulate events over 33ms per frame
  std::uint64_t event_accumulation_time = 33000;

  void paintFrame();

  template <typename T>
  std::size_t binarySearch(const std::vector<T> &data, std::uint64_t time,
                           std::size_t min, std::size_t max) const;
};
}

#endif // LABELER_DAVIS_WINDOW_H_
