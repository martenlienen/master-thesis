#include <cstdint>
#include <stdexcept>

#include "AedatStorage.h"

namespace recorder {

namespace store {

AedatStorage::AedatStorage(std::string path)
    : stream(path.c_str(), std::ios::out | std::ios::binary) {
  if (!this->stream.is_open()) {
    throw std::runtime_error("Could not open aedat file");
  }

  this->stream << "#!AER-DAT1.0" << CRLF;
}

void AedatStorage::write(edvs_event_t event) {
  uint32_t timestamp = event.t;
  uint16_t address = (event.y << 8) | (event.x << 1) | event.parity;

  // Write bytes in big-endian order
  this->stream.write(reinterpret_cast<char *>(&address) + 1, 1);
  this->stream.write(reinterpret_cast<char *>(&address) + 0, 1);
  this->stream.write(reinterpret_cast<char *>(&timestamp) + 3, 1);
  this->stream.write(reinterpret_cast<char *>(&timestamp) + 2, 1);
  this->stream.write(reinterpret_cast<char *>(&timestamp) + 1, 1);
  this->stream.write(reinterpret_cast<char *>(&timestamp) + 0, 1);
}
}
}
