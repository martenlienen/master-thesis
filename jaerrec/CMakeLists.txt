cmake_minimum_required(VERSION 3.7)

project(JAERREC)

# Include header-only libraries
include_directories(libs)

find_package(wxWidgets COMPONENTS core base media REQUIRED)
include(${wxWidgets_USE_FILE})

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

set(JAERREC_SRCS src/main.cpp src/App.cpp src/MainFrame.cpp
  src/ConnectDialog.cpp)

add_executable(jaerrec ${JAERREC_SRCS})
set_property(TARGET jaerrec PROPERTY CXX_STANDARD 14)
target_link_libraries(jaerrec ${wxWidgets_LIBRARIES} ${OpenCV_LIBS}
  ${Boost_LIBRARIES})
