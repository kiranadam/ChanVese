cmake_minimum_required(VERSION 2.8) 
PROJECT(ChanVese)
find_package(OpenCV REQUIRED)
set(NAME_SRC
    src/ChanVese.cpp
    src/test_main.cpp
)

set(NAME_HEADERS       
    include/ChanVese.hpp
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(ChanVese ${NAME_SRC} ${NAME_HEADERS})

target_link_libraries(ChanVese ${OpenCV_LIBS})
