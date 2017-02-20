#include "TimestampFile.h"

namespace recorder {

namespace agents {

TimestampFile::TimestampFile(std::string path) : stream(path) {
  if (!this->stream.is_open()) {
    throw std::runtime_error("Could not open timestamp file " + path);
  }

  stream << "id,start,end\n";
}

void TimestampFile::pushTimestamp(std::string name, uint64_t start,
                                  uint64_t end) {
  this->stream << name << ",";
  this->stream << start << ",";
  this->stream << end << "\n";
}
}
}
