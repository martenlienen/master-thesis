cmake_minimum_required(VERSION 3.7)

project(RECORDER)

find_package(wxWidgets COMPONENTS core base media REQUIRED)
include(${wxWidgets_USE_FILE})

set(RECORDER_SRCS src/main.cpp src/App.cpp src/gui/Instructor.cpp)

add_executable(recorder ${RECORDER_SRCS})
set_property(TARGET recorder PROPERTY CXX_STANDARD 14)
target_link_libraries(recorder ${wxWidgets_LIBRARIES})
