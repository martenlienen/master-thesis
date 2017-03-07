cmake_minimum_required(VERSION 3.7)

project(LABELER)

# Include header-only libraries
include_directories(libs)

find_package(wxWidgets COMPONENTS core base media REQUIRED)
include(${wxWidgets_USE_FILE})

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

set(LABELER_SRCS src/main.cpp src/App.cpp src/MainFrame.cpp src/DAVISWindow.cpp
  src/PlayTimer.cpp)

add_executable(labeler ${LABELER_SRCS})
set_property(TARGET labeler PROPERTY CXX_STANDARD 14)
target_link_libraries(labeler ${wxWidgets_LIBRARIES} ${OpenCV_LIBS}
  ${Boost_LIBRARIES})
