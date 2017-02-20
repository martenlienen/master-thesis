#ifndef RECORDER_AGENTS_TIMESTAMP_FILE_H_
#define RECORDER_AGENTS_TIMESTAMP_FILE_H_

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace recorder {

namespace agents {

class TimestampFile {
public:
  TimestampFile(std::string path);

  void pushTimestamp(std::string name, uint64_t start, uint64_t end);

private:
  std::ofstream stream;
};
}
}

#endif // RECORDER_AGENTS_TIMESTAMP_FILE_H_
