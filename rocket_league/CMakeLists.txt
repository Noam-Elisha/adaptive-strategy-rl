cmake_minimum_required(VERSION 3.8)

project("RocketLeagueStrategyBot")

# Glob our own source files
file(GLOB_RECURSE BOT_SRC "src/*.cpp" "src/*.h" "src/*.hpp")

# Include RLBotClient from the GigaLearnCPP-Leak project (RLBot integration)
set(RLBOT_CLIENT_SRC
    "GigaLearnCPP-Leak/src/RLBotClient.cpp"
    "GigaLearnCPP-Leak/src/RLBotClient.h"
)

add_executable(RocketLeagueStrategyBot ${BOT_SRC} ${RLBOT_CLIENT_SRC})

# C++20
set_target_properties(RocketLeagueStrategyBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RocketLeagueStrategyBot PROPERTIES CXX_STANDARD 20)

# Make sure libs are output in the same directory as the executable
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include GigaLearnCPP (which pulls in RLGymCPP, RocketSim, LibTorch, etc.)
add_subdirectory(GigaLearnCPP-Leak/GigaLearnCPP)
target_link_libraries(RocketLeagueStrategyBot GigaLearnCPP)

# Include RLBotCPP (TCP server, bot manager for RLBot integration)
add_subdirectory(GigaLearnCPP-Leak/RLBotCPP)
target_link_libraries(RocketLeagueStrategyBot RLBotCPP)

# Our src/ headers + GigaLearnCPP-Leak/src/ (for RLBotClient.h)
target_include_directories(RocketLeagueStrategyBot PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/GigaLearnCPP-Leak/src"
)
