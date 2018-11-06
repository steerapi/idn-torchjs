cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Name of the project (will be the name of the plugin)
project (torchjs)

find_package(CUDA)
if(CUDA_FOUND)
  file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-shared-with-deps-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch.zip)
else()
  file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch.zip)
endif(CUDA_FOUND)

execute_process(
  COMMAND unzip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch.zip
)

execute_process(
  COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/libtorch.zip
)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/libtorch/)

find_package(Torch REQUIRED)

# Essential include files to build a node addon,
# you should add this line in every CMake.js based project.
include_directories(${CMAKE_JS_INC})

# Declare the location of the source files
file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")

# This line will tell CMake that we're building a shared library
# from the above source files
# named after the project's name
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

# This line will give our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Essential library files to link to a node addon,
# you should add this line in every CMake.js based project.
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB};${TORCH_LIBRARIES})
