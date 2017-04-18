#include <cstdint>
#include <string>
#include <vector>

#include <wx/dialog.h>

namespace jaerrec {

class ConnectDialog : public wxDialog {
public:
  std::string CACHE_PATH = "/tmp/jaerrec-cache";

  std::string subject = "";
  std::string ip = "";
  std::uint16_t port = 8997;
  std::string path = "";
  std::string instruction_dir = "";
  std::vector<std::string> gestures;

  ConnectDialog(wxWindow *parent);

private:
  void readCacheFile();
  void writeCacheFile();
};
}
