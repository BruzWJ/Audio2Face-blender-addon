cmake_minimum_required(VERSION 3.24)

project(audio2face_worker VERSION 0.1.0 LANGUAGES CXX)

set(A2F_SDK_SOURCE_DIR "" CACHE PATH
  "Pinned NVIDIA Audio2Face-3D-SDK 1.0.0 source directory")

if(A2F_SDK_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR
    "A2F_SDK_SOURCE_DIR must name a pinned NVIDIA Audio2Face-3D-SDK 1.0.0 source tree")
endif()
get_filename_component(A2F_SDK_SOURCE_DIR "${A2F_SDK_SOURCE_DIR}" ABSOLUTE)
foreach(_a2f_sdk_input IN ITEMS
    CMakeLists.txt
    audio2x-sdk/CMakeLists.txt
    audio2x-sdk/VERSION.md
    LICENSE.txt)
  if(NOT EXISTS "${A2F_SDK_SOURCE_DIR}/${_a2f_sdk_input}" OR
     IS_DIRECTORY "${A2F_SDK_SOURCE_DIR}/${_a2f_sdk_input}")
    message(FATAL_ERROR
      "A2F_SDK_SOURCE_DIR is not a complete Audio2Face-3D-SDK source tree: "
      "${_a2f_sdk_input} is missing")
  endif()
endforeach()
file(STRINGS "${A2F_SDK_SOURCE_DIR}/audio2x-sdk/VERSION.md"
  _a2f_sdk_version LIMIT_COUNT 1)
string(STRIP "${_a2f_sdk_version}" _a2f_sdk_version)
if(NOT _a2f_sdk_version STREQUAL "1.0.0")
  message(FATAL_ERROR
    "The worker requires Audio2Face-3D-SDK 1.0.0; got ${_a2f_sdk_version}")
endif()

# The SDK is a source-only CMake project. Its concrete shared-library target is
# created by audio2x-sdk/CMakeLists.txt; it does not publish an install package.
add_subdirectory("${A2F_SDK_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/audio2face-3d-sdk" EXCLUDE_FROM_ALL)
if(NOT TARGET audio2x)
  message(FATAL_ERROR
    "Audio2Face-3D-SDK 1.0.0 did not define audio2x")
endif()

# FindNlohmannJson.cmake creates a directory-scoped imported target inside the
# SDK source tree. Consume the exact SDK-owned Packman dependency instead of
# performing another package search from the parent project.
set(_a2f_nlohmann_include
  "${A2F_SDK_SOURCE_DIR}/_deps/target-deps/nlohmann_json/include")
if(NOT EXISTS "${_a2f_nlohmann_include}/nlohmann/json.hpp" OR
   IS_DIRECTORY "${_a2f_nlohmann_include}/nlohmann/json.hpp")
  message(FATAL_ERROR
    "Audio2Face-3D-SDK 1.0.0 did not resolve nlohmann/json.hpp")
endif()

set(A2F_RUNTIME_OUTPUT_DIR "" CACHE PATH
  "Final platform runtime directory populated by the release builder")

add_executable(audio2face_worker
  src/a2f_backend.cpp
  src/main.cpp
  src/path_contract.cpp
  src/protocol.cpp
  src/result_file.cpp
  src/wav.cpp
)

target_compile_features(audio2face_worker PRIVATE cxx_std_17)
target_compile_definitions(audio2face_worker PRIVATE
  A2F_WORKER_VERSION="${PROJECT_VERSION}"
)
target_include_directories(audio2face_worker PRIVATE
  "${_a2f_nlohmann_include}"
)
target_link_libraries(audio2face_worker PRIVATE
  audio2x)

if(MSVC)
  target_compile_options(audio2face_worker PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(audio2face_worker PRIVATE -Wall -Wextra -Wpedantic)
  find_package(Threads REQUIRED)
  target_link_libraries(audio2face_worker PRIVATE Threads::Threads)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _a2f_system_processor)
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND WIN32 AND
   _a2f_system_processor STREQUAL "amd64")
  set(A2F_RUNTIME_PLATFORM "windows-x64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8 AND
       CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
       _a2f_system_processor STREQUAL "x86_64")
  set(A2F_RUNTIME_PLATFORM "linux-x64")
else()
  message(FATAL_ERROR
    "The bundled runtime supports only Windows x64 and Linux x64; "
    "got ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(A2F_RUNTIME_OUTPUT_DIR STREQUAL "" OR
   NOT IS_ABSOLUTE "${A2F_RUNTIME_OUTPUT_DIR}")
  message(FATAL_ERROR
    "A2F_RUNTIME_OUTPUT_DIR must be an absolute release output directory")
endif()
get_filename_component(A2F_RUNTIME_OUTPUT_DIR
  "${A2F_RUNTIME_OUTPUT_DIR}" ABSOLUTE)

set_target_properties(audio2face_worker PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${A2F_RUNTIME_OUTPUT_DIR}/bin")
set_target_properties(audio2x PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/import")
if(MSVC)
  set_target_properties(audio2face_worker audio2x PROPERTIES
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/symbols"
    PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/symbols")
endif()
if(WIN32)
  set_target_properties(audio2x PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${A2F_RUNTIME_OUTPUT_DIR}/bin")
else()
  set_target_properties(audio2x PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${A2F_RUNTIME_OUTPUT_DIR}/lib")
endif()

message(STATUS
  "Audio2Face ${A2F_RUNTIME_PLATFORM} native outputs: ${A2F_RUNTIME_OUTPUT_DIR}")
