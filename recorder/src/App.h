#include <memory>

#include <wx/wx.h>

namespace recorder {

class App : public wxApp {
public:
  virtual bool OnInit();

private:
  std::unique_ptr<wxFrame> controller;
  std::unique_ptr<wxFrame> instructor;
  std::unique_ptr<wxFrame> feedback;
};
}
