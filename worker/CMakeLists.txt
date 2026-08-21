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

# The Python runtime contract supplies the sole package-path map. CMake builds
# the two generated targets and copies the declared external source list.
set(A2F_STAGE_WORKER_PATH "" CACHE STRING
  "Contract path for the generated Audio2Face worker")
set(A2F_STAGE_AUDIO2X_PATH "" CACHE STRING
  "Contract path for the generated Audio2Face SDK library")
set(A2F_STAGE_TRTEXEC_PATH "" CACHE STRING
  "Contract path for the bundled TensorRT trtexec executable")
set(A2F_STAGE_EXTERNAL_SOURCES "" CACHE STRING
  "Ordered source files supplied by the Python runtime contract")
set(A2F_STAGE_EXTERNAL_PATHS "" CACHE STRING
  "Ordered package paths supplied by the Python runtime contract")

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

foreach(_a2f_contract_value IN ITEMS
    A2F_STAGE_WORKER_PATH
    A2F_STAGE_AUDIO2X_PATH
    A2F_STAGE_TRTEXEC_PATH
    A2F_STAGE_EXTERNAL_SOURCES
    A2F_STAGE_EXTERNAL_PATHS)
  if("${${_a2f_contract_value}}" STREQUAL "")
    message(FATAL_ERROR
      "The Python runtime contract did not supply ${_a2f_contract_value}")
  endif()
endforeach()
list(LENGTH A2F_STAGE_EXTERNAL_SOURCES _a2f_source_count)
list(LENGTH A2F_STAGE_EXTERNAL_PATHS _a2f_path_count)
if(NOT _a2f_source_count EQUAL _a2f_path_count)
  message(FATAL_ERROR
    "The runtime contract source/path counts differ: "
    "${_a2f_source_count} sources, ${_a2f_path_count} paths")
endif()

set(_a2f_runtime_root "${CMAKE_BINARY_DIR}/runtime")
set(_a2f_stage_config_directory "${CMAKE_CURRENT_BINARY_DIR}/runtime-stage-config")
file(MAKE_DIRECTORY "${_a2f_stage_config_directory}")
set(_a2f_stage_config
  "${_a2f_stage_config_directory}/StageRuntime-$<CONFIG>.cmake")
set(_a2f_stage_config_content [====[
set(A2F_STAGE_ROOT [==[@A2F_STAGE_ROOT@]==])
set(A2F_PLATFORM [==[@A2F_PLATFORM@]==])
set(A2F_WORKER_FILE [==[@A2F_WORKER_FILE@]==])
set(A2F_AUDIO2X_FILE [==[@A2F_AUDIO2X_FILE@]==])
set(A2F_WORKER_PATH [==[@A2F_WORKER_PATH@]==])
set(A2F_AUDIO2X_PATH [==[@A2F_AUDIO2X_PATH@]==])
set(A2F_TRTEXEC_PATH [==[@A2F_TRTEXEC_PATH@]==])
set(A2F_EXTERNAL_SOURCES [==[@A2F_EXTERNAL_SOURCES@]==])
set(A2F_EXTERNAL_PATHS [==[@A2F_EXTERNAL_PATHS@]==])
]====])
set(A2F_STAGE_ROOT "${_a2f_runtime_root}")
set(A2F_PLATFORM "${A2F_RUNTIME_PLATFORM}")
set(A2F_WORKER_FILE "$<TARGET_FILE:audio2face_worker>")
set(A2F_AUDIO2X_FILE "$<TARGET_FILE:audio2x>")
set(A2F_WORKER_PATH "${A2F_STAGE_WORKER_PATH}")
set(A2F_AUDIO2X_PATH "${A2F_STAGE_AUDIO2X_PATH}")
set(A2F_TRTEXEC_PATH "${A2F_STAGE_TRTEXEC_PATH}")
set(A2F_EXTERNAL_SOURCES "${A2F_STAGE_EXTERNAL_SOURCES}")
set(A2F_EXTERNAL_PATHS "${A2F_STAGE_EXTERNAL_PATHS}")
string(CONFIGURE "${_a2f_stage_config_content}" _a2f_stage_config_content @ONLY)
file(GENERATE
  OUTPUT "${_a2f_stage_config}"
  CONTENT "${_a2f_stage_config_content}"
  TARGET audio2face_worker)

add_custom_target(audio2face_runtime_stage
  COMMAND "${CMAKE_COMMAND}"
    "-DA2F_STAGE_CONFIG=${_a2f_stage_config}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageRuntime.cmake"
  DEPENDS audio2face_worker
  COMMENT "Staging contract-defined ${A2F_RUNTIME_PLATFORM} Audio2Face runtime"
  VERBATIM)

message(STATUS
  "Audio2Face runtime staging enabled: build target "
  "audio2face_runtime_stage -> "
  "${_a2f_runtime_root}/${A2F_RUNTIME_PLATFORM}")
