
# Set include directories
set(C_EMBED_SOM_INCLUDE_DIRS
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
  GLOB_RECURSE C_EMBED_SOM_SRC_FILEPATHS
    "*.h"
    "*.cpp"
)

# Stup include dirs
include_directories(${C_EMBED_SOM_INCLUDE_DIRS})

#############################
# Create main executable
#############################
add_executable(
  ${MAIN_EXE_NAME} 
  ${SYENGINE_SOURCE_MAIN} ${C_EMBED_SOM_SRC_FILEPATHS}
)
INSTALL_TARGETS(T ${MAIN_EXE_NAME})

if(WIN32)
  add_compile_definitions(NOMINMAX)
  include_directories(${CMAKE_SOURCE_DIR}/external/win64/igraph-0.7.1/include)
  target_link_libraries(${MAIN_EXE_NAME}  ${CMAKE_SOURCE_DIR}/external/win64/igraph-0.7.1/lib/igraph.lib)
else()
  
endif(WIN32)
