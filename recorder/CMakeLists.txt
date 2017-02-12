cmake_minimum_required(VERSION 3.7)

project(RECORDER)

# Include header-only libraries
include_directories(libs)

# Compile the edvs library from edvstools
add_subdirectory(Edvs)
include_directories(Edvs)

find_package(wxWidgets COMPONENTS core base media REQUIRED)
include(${wxWidgets_USE_FILE})

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

set(RECORDER_SRCS src/main.cpp src/App.cpp src/gui/Setup.cpp
  src/gui/Controller.cpp src/gui/Instructor.cpp src/gui/OpenCVFrame.cpp
  src/gui/DVSFrame.cpp src/gui/OpenCVDisplay.cpp src/gui/DVSDisplay.cpp
  src/capture/OpenCVCapture.cpp src/capture/DVSCapture.cpp
  src/store/AedatStorage.cpp src/store/VideoStorage.cpp src/agents/DVSAgent.cpp
  src/agents/OpenCVAgent.cpp)

add_executable(recorder ${RECORDER_SRCS})
set_property(TARGET recorder PROPERTY CXX_STANDARD 14)
target_link_libraries(recorder ${wxWidgets_LIBRARIES} ${OpenCV_LIBS}
  ${Boost_LIBRARIES} Edvs)
