#ifndef RECORDER_STORE_AEDAT_STORAGE_H_
#define RECORDER_STORE_AEDAT_STORAGE_H_

#include <fstream>
#include <string>

extern "C" {
#include "edvs.h"
}

namespace recorder {

namespace store {

class AedatStorage {
public:
  const std::string CRLF = "\r\n";

  AedatStorage(std::string path);

  void write(edvs_event_t event);

private:
  std::ofstream stream;
};
}
}

#endif // RECORDER_STORE_AEDAT_STORAGE_H_
