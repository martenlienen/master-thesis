#ifndef LABELER_STREAM_H_
#define LABELER_STREAM_H_

#include <opencv2/core/core.hpp>

namespace labeler {

struct StreamFrame {
  std::uint64_t timestamp;
  cv::Mat frame;
};

struct StreamEvent {
  std::uint64_t timestamp;
  std::uint16_t x;
  std::uint16_t y;
  std::uint8_t parity;
};

struct StreamLabel {
  std::uint64_t start;
  std::uint64_t end;
  std::string label;
};
}

#endif // LABELER_STREAM_H_
