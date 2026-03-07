cmake_minimum_required(VERSION 3.8)

project("RocketLeagueStrategyBot")

# Glob our own source files
file(GLOB_RECURSE BOT_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(RocketLeagueStrategyBot ${BOT_SRC})

# C++20
set_target_properties(RocketLeagueStrategyBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketLeagueStrategyBot PROPERTIES CXX_STANDARD 20)

# Make sure libs are output in the same directory as the executable
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include GigaLearnCPP (which pulls in RLGymCPP, RocketSim, LibTorch, etc.)
add_subdirectory(GigaLearnCPP-Leak/GigaLearnCPP)
target_link_libraries(RocketLeagueStrategyBot GigaLearnCPP)

# Our src/ headers need to be findable
target_include_directories(RocketLeagueStrategyBot PRIVATE "${PROJECT_SOURCE_DIR}/src")
