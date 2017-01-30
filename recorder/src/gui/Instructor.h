#include <wx/frame.h>
#include <wx/mediactrl.h>

namespace recorder {

namespace gui {

class Instructor : public wxFrame {
public:
  Instructor();

private:
  wxMediaCtrl *player;
};
}
}
