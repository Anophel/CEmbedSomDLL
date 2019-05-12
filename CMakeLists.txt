
# Create global project name variable
set(PROJECT_NAME CEmbedSom)
set(MAIN_EXE_NAME CEmbedSom)

# Set project name
project(${PROJECT_NAME})
cmake_minimum_required(VERSION 3.8)


# Set C++17
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -lstdc++fs")
endif(MSVC) 


set(CMAKE_INSTALL_PREFIX "./dist")


# Set version
set(CMAKE_PROJECT_VERSION_MAJOR 0)
set(CMAKE_PROJECT_VERSION_MINOR 0)
set(CMAKE_PROJECT_VERSION_PATCH 1)
set(CMAKE_PROJECT_VERSION ${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH})

# Set the correct macro directory path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    

# Add project
add_subdirectory(src)