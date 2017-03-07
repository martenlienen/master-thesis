#include "PlayTimer.h"

namespace labeler {

PlayTimer::PlayTimer(std::function<void(std::int64_t)> cb) : cb(cb) {}

void PlayTimer::setSpeed(float speed) { this->speed = speed; }

bool PlayTimer::Start(int milliseconds, bool oneshot) {
  this->trigger_time = std::chrono::high_resolution_clock::now();

  return wxTimer::Start(milliseconds, oneshot);
}

void PlayTimer::Notify() {
  auto prev_time = this->trigger_time;
  this->trigger_time = std::chrono::high_resolution_clock::now();
  auto delta = this->trigger_time - prev_time;
  auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(delta).count();

  microseconds *= this->speed;

  this->cb(microseconds);
}
}
