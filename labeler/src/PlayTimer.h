#ifndef LABELER_PLAY_TIMER_H_
#define LABELER_PLAY_TIMER_H_

#include <chrono>
#include <cstdint>
#include <functional>

#include <wx/timer.h>

namespace labeler {

/**
 * Triggers a callback every few milliseconds and tells it how many microseconds
 * have passed in video-time.
 */
class PlayTimer : public wxTimer {
  // Time of the last trigger
  std::chrono::time_point<std::chrono::high_resolution_clock> trigger_time;
  const std::function<void(std::int64_t)> cb;
  float speed = 1.0;

public:
  PlayTimer(std::function<void(std::int64_t)> cb);

  void setSpeed(float speed);

  virtual bool Start(int milliseconds = -1, bool oneshot = wxTIMER_CONTINUOUS);
  virtual void Notify();
};
}

#endif // LABELER_PLAY_TIMER_H_
