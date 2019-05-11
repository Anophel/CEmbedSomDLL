
# Set include directories
set(SYENGINE_INCLUDE_DIRS
  ../include
  .
)

# Get main source file
file( 
  GLOB SYENGINE_SOURCE_MAIN
  "../main.cpp"
)

# Get all header and cpp files
file(
  GLOB_RECURSE SYENGINE_SOURCE_FILEPATHS
    "*.h"
    "*.cpp"
)

# Stup include dirs
include_directories(${SYENGINE_INCLUDE_DIRS})

#############################
# Create engine executable
#############################
add_executable(
  ${MAIN_EXE_NAME} 
  ${SYENGINE_SOURCE_MAIN} ${SYENGINE_SOURCE_FILEPATHS}
)
INSTALL_TARGETS(T ${MAIN_EXE_NAME})

if(WIN32)
  add_compile_definitions(NOMINMAX)
else()
  
endif(WIN32)
