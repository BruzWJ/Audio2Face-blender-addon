cmake_minimum_required(VERSION 3.24)

project(a2f_blender_worker VERSION 0.1.0 LANGUAGES CXX)

option(A2F_WORKER_STAGE_RUNTIME
  "Enable strict production runtime staging and archive targets" OFF)
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

# The SDK is a source-only CMake project. Its documented consumer target is
# created by audio2x-sdk/CMakeLists.txt; it does not publish an install package.
add_subdirectory("${A2F_SDK_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/audio2face-3d-sdk" EXCLUDE_FROM_ALL)
foreach(_a2f_sdk_target IN ITEMS
    audio2x-sdk::audio2x
    nlohmann_json::nlohmann_json)
  if(NOT TARGET ${_a2f_sdk_target})
    message(FATAL_ERROR
      "Audio2Face-3D-SDK 1.0.0 did not define ${_a2f_sdk_target}")
  endif()
endforeach()

# Release-runtime inputs. Staging only uses these exact, reviewed paths.
set(A2F_CUDA_RUNTIME_DIR "" CACHE PATH
  "Directory containing the reviewed CUDA runtime libraries for redistribution")
set(A2F_TENSORRT_RUNTIME_DIR "" CACHE PATH
  "Directory containing the reviewed TensorRT runtime libraries for redistribution")
set(A2F_BUNDLE_TRTEXEC "" CACHE FILEPATH
  "Release-built NVIDIA TensorRT trtexec executable for first-run engine creation")
set(A2F_BUNDLE_TRTEXEC_SOURCE_LICENSE "" CACHE FILEPATH
  "License for the pinned NVIDIA TensorRT source used to build bundled trtexec")
set(A2F_BUNDLE_TRTEXEC_PROVENANCE "" CACHE FILEPATH
  "Pinned NVIDIA TensorRT tag/commit and bundled trtexec build provenance")
set(A2F_BUNDLE_AUDIO2FACE_MODEL_DIR "" CACHE PATH
  "Official Audio2Face-3D v3.0 ONNX model directory to bundle")
set(A2F_BUNDLE_AUDIO2FACE_MODEL_LICENSE "" CACHE FILEPATH
  "NVIDIA Audio2Face model license file to include in the runtime archive")
set(A2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR "" CACHE PATH
  "Official Audio2Emotion v3.0 ONNX model directory to bundle")
set(A2F_BUNDLE_AUDIO2EMOTION_MODEL_LICENSE "" CACHE FILEPATH
  "NVIDIA Audio2Emotion model license file to include in the runtime archive")

add_executable(a2f_blender_worker
  src/a2f_backend.cpp
  src/main.cpp
  src/protocol.cpp
  src/result_file.cpp
  src/wav.cpp
)

target_compile_features(a2f_blender_worker PRIVATE cxx_std_17)
target_compile_definitions(a2f_blender_worker PRIVATE
  A2F_WORKER_VERSION="${PROJECT_VERSION}"
)
target_include_directories(a2f_blender_worker PRIVATE include)
target_link_libraries(a2f_blender_worker PRIVATE
  audio2x-sdk::audio2x
  nlohmann_json::nlohmann_json)

if(MSVC)
  target_compile_options(a2f_blender_worker PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(a2f_blender_worker PRIVATE -Wall -Wextra -Wpedantic)
  find_package(Threads REQUIRED)
  target_link_libraries(a2f_blender_worker PRIVATE Threads::Threads)
endif()

if(A2F_WORKER_STAGE_RUNTIME)
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
      "Runtime archives currently support only Windows x64 and Linux x64; "
      "got ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  function(a2f_require_release_file variable description)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
      message(FATAL_ERROR
        "${description} is required when A2F_WORKER_STAGE_RUNTIME=ON "
        "(-D${variable}=<absolute path>)")
    endif()
    get_filename_component(_a2f_file "${${variable}}" ABSOLUTE)
    if(NOT EXISTS "${_a2f_file}" OR IS_DIRECTORY "${_a2f_file}")
      message(FATAL_ERROR "${description} is not a file: ${_a2f_file}")
    endif()
    set(${variable} "${_a2f_file}" PARENT_SCOPE)
  endfunction()

  function(a2f_require_release_directory variable description)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
      message(FATAL_ERROR
        "${description} is required when A2F_WORKER_STAGE_RUNTIME=ON "
        "(-D${variable}=<absolute path>)")
    endif()
    get_filename_component(_a2f_directory "${${variable}}" ABSOLUTE)
    if(NOT IS_DIRECTORY "${_a2f_directory}")
      message(FATAL_ERROR "${description} is not a directory: ${_a2f_directory}")
    endif()
    set(${variable} "${_a2f_directory}" PARENT_SCOPE)
  endfunction()

  function(a2f_require_runtime output directory filename description)
    set(_a2f_runtime "${directory}/${filename}")
    if(NOT EXISTS "${_a2f_runtime}" OR IS_DIRECTORY "${_a2f_runtime}")
      message(FATAL_ERROR
        "Missing ${description}: ${_a2f_runtime}")
    endif()
    get_filename_component(_a2f_runtime "${_a2f_runtime}" ABSOLUTE)
    set(${output} "${_a2f_runtime}" PARENT_SCOPE)
  endfunction()

  a2f_require_release_directory(A2F_CUDA_RUNTIME_DIR "CUDA runtime directory")
  a2f_require_release_directory(A2F_TENSORRT_RUNTIME_DIR "TensorRT runtime directory")
  a2f_require_release_directory(A2F_BUNDLE_AUDIO2FACE_MODEL_DIR
    "Audio2Face-3D v3.0 model directory")
  a2f_require_release_directory(A2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR
    "Audio2Emotion v3.0 model directory")
  a2f_require_release_file(A2F_BUNDLE_TRTEXEC
    "Release-built NVIDIA TensorRT trtexec executable")
  a2f_require_release_file(A2F_BUNDLE_TRTEXEC_SOURCE_LICENSE
    "Pinned NVIDIA TensorRT source license")
  a2f_require_release_file(A2F_BUNDLE_TRTEXEC_PROVENANCE
    "Pinned NVIDIA TensorRT trtexec build provenance")
  a2f_require_release_file(A2F_BUNDLE_AUDIO2FACE_MODEL_LICENSE
    "Audio2Face model license")
  a2f_require_release_file(A2F_BUNDLE_AUDIO2EMOTION_MODEL_LICENSE
    "Audio2Emotion model license")

  set(_a2f_sdk_license "${A2F_SDK_SOURCE_DIR}/LICENSE.txt")
  set(_a2f_cuda_license "${A2F_SDK_SOURCE_DIR}/licenses/cuda-LICENSE.txt")
  set(_a2f_tensorrt_license "${A2F_SDK_SOURCE_DIR}/licenses/TensorRT-Readme.txt")
  set(_a2f_tensorrt_acknowledgements
    "${A2F_SDK_SOURCE_DIR}/licenses/TensorRT-Acknowledgements.txt")
  foreach(_a2f_sdk_notice IN ITEMS
      _a2f_sdk_license
      _a2f_cuda_license
      _a2f_tensorrt_license
      _a2f_tensorrt_acknowledgements)
    a2f_require_release_file(${_a2f_sdk_notice} "Audio2Face SDK notice")
  endforeach()

  function(a2f_validate_release_model directory label variable_name)
    foreach(_a2f_model_file model.json network.onnx trt_info.json)
      if(NOT EXISTS "${directory}/${_a2f_model_file}" OR
         IS_DIRECTORY "${directory}/${_a2f_model_file}")
        message(FATAL_ERROR
          "Official ${label} ONNX model input is incomplete: "
          "${directory}/${_a2f_model_file} is missing")
      endif()
    endforeach()
    file(GLOB_RECURSE _a2f_model_inputs LIST_DIRECTORIES FALSE
      "${directory}/*")
    set(_a2f_prebuilt_engines)
    foreach(_a2f_model_input IN LISTS _a2f_model_inputs)
      get_filename_component(_a2f_model_extension "${_a2f_model_input}" LAST_EXT)
      string(TOLOWER "${_a2f_model_extension}" _a2f_model_extension)
      if(_a2f_model_extension STREQUAL ".trt" OR
         _a2f_model_extension STREQUAL ".engine")
        list(APPEND _a2f_prebuilt_engines "${_a2f_model_input}")
      endif()
    endforeach()
    if(_a2f_prebuilt_engines)
      message(FATAL_ERROR
        "The downloadable runtime must be GPU-neutral and cannot contain "
        "prebuilt TensorRT engines. Remove these files from ${variable_name}: "
        "${_a2f_prebuilt_engines}")
    endif()
  endfunction()

  a2f_validate_release_model(
    "${A2F_BUNDLE_AUDIO2FACE_MODEL_DIR}"
    "Audio2Face-3D v3.0"
    "A2F_BUNDLE_AUDIO2FACE_MODEL_DIR")
  a2f_validate_release_model(
    "${A2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR}"
    "Audio2Emotion v3.0"
    "A2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR")

  set(_a2f_runtime_files "$<TARGET_FILE:audio2x-sdk::audio2x>")

  if(WIN32)
    a2f_require_runtime(_a2f_cudart "${A2F_CUDA_RUNTIME_DIR}"
      cudart64_12.dll "CUDA runtime")
    a2f_require_runtime(_a2f_cublas "${A2F_CUDA_RUNTIME_DIR}"
      cublas64_12.dll "cuBLAS runtime")
    a2f_require_runtime(_a2f_cublas_lt "${A2F_CUDA_RUNTIME_DIR}"
      cublasLt64_12.dll "cuBLAS Lt runtime")
    a2f_require_runtime(_a2f_curand "${A2F_CUDA_RUNTIME_DIR}"
      curand64_10.dll "cuRAND runtime")
    a2f_require_runtime(_a2f_nvinfer "${A2F_TENSORRT_RUNTIME_DIR}"
      nvinfer_10.dll "TensorRT inference runtime")
    a2f_require_runtime(_a2f_nvinfer_plugin "${A2F_TENSORRT_RUNTIME_DIR}"
      nvinfer_plugin_10.dll "TensorRT plugin runtime")
    a2f_require_runtime(_a2f_nvonnxparser "${A2F_TENSORRT_RUNTIME_DIR}"
      nvonnxparser_10.dll "TensorRT ONNX parser runtime")
    a2f_require_runtime(_a2f_nvinfer_builder "${A2F_TENSORRT_RUNTIME_DIR}"
      nvinfer_builder_resource_10.dll "TensorRT builder resource runtime")
  else()
    a2f_require_runtime(_a2f_cudart "${A2F_CUDA_RUNTIME_DIR}"
      libcudart.so.12 "CUDA runtime")
    a2f_require_runtime(_a2f_cublas "${A2F_CUDA_RUNTIME_DIR}"
      libcublas.so.12 "cuBLAS runtime")
    a2f_require_runtime(_a2f_cublas_lt "${A2F_CUDA_RUNTIME_DIR}"
      libcublasLt.so.12 "cuBLAS Lt runtime")
    a2f_require_runtime(_a2f_curand "${A2F_CUDA_RUNTIME_DIR}"
      libcurand.so.10 "cuRAND runtime")
    a2f_require_runtime(_a2f_nvinfer "${A2F_TENSORRT_RUNTIME_DIR}"
      libnvinfer.so.10 "TensorRT inference runtime")
    a2f_require_runtime(_a2f_nvinfer_plugin "${A2F_TENSORRT_RUNTIME_DIR}"
      libnvinfer_plugin.so.10 "TensorRT plugin runtime")
    a2f_require_runtime(_a2f_nvonnxparser "${A2F_TENSORRT_RUNTIME_DIR}"
      libnvonnxparser.so.10 "TensorRT ONNX parser runtime")
    a2f_require_runtime(_a2f_nvinfer_builder "${A2F_TENSORRT_RUNTIME_DIR}"
      libnvinfer_builder_resource.so.10 "TensorRT builder resource runtime")

    # The copied worker must locate the sibling lib directory without any
    # build-host absolute RPATH. The add-on also supplies a child-only
    # LD_LIBRARY_PATH so dependencies loaded by audio2x/TensorRT resolve there.
    set_target_properties(a2f_blender_worker PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN/../lib")
  endif()

  list(APPEND _a2f_runtime_files
    "${_a2f_cudart}"
    "${_a2f_cublas}"
    "${_a2f_cublas_lt}"
    "${_a2f_curand}"
    "${_a2f_nvinfer}"
    "${_a2f_nvinfer_plugin}"
    "${_a2f_nvonnxparser}"
    "${_a2f_nvinfer_builder}")

  set(_a2f_runtime_root "${CMAKE_BINARY_DIR}/runtime")

  set(_a2f_project_license "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
  set(_a2f_third_party_notices "${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md")
  foreach(_a2f_project_notice "${_a2f_project_license}" "${_a2f_third_party_notices}")
    if(NOT EXISTS "${_a2f_project_notice}")
      message(FATAL_ERROR "Required project notice is missing: ${_a2f_project_notice}")
    endif()
  endforeach()

  set(_a2f_stage_config_directory "${CMAKE_CURRENT_BINARY_DIR}/runtime-stage-config")
  file(MAKE_DIRECTORY "${_a2f_stage_config_directory}")
  set(_a2f_stage_config
    "${_a2f_stage_config_directory}/StageRuntime-$<CONFIG>.cmake")
  set(_a2f_stage_config_content [====[
set(A2F_STAGE_ROOT [==[@A2F_STAGE_ROOT@]==])
set(A2F_PLATFORM [==[@A2F_PLATFORM@]==])
set(A2F_PACKAGE_VERSION [==[@A2F_PACKAGE_VERSION@]==])
set(A2F_WORKER_FILE [==[@A2F_WORKER_FILE@]==])
set(A2F_RUNTIME_FILES [==[@A2F_RUNTIME_FILES@]==])
set(A2F_TRTEXEC_FILE [==[@A2F_TRTEXEC_FILE@]==])
set(A2F_TRTEXEC_SOURCE_LICENSE [==[@A2F_TRTEXEC_SOURCE_LICENSE@]==])
set(A2F_TRTEXEC_PROVENANCE [==[@A2F_TRTEXEC_PROVENANCE@]==])
set(A2F_AUDIO2FACE_MODEL_DIR [==[@A2F_AUDIO2FACE_MODEL_DIR@]==])
set(A2F_AUDIO2EMOTION_MODEL_DIR [==[@A2F_AUDIO2EMOTION_MODEL_DIR@]==])
set(A2F_PROJECT_LICENSE [==[@A2F_PROJECT_LICENSE@]==])
set(A2F_THIRD_PARTY_NOTICES [==[@A2F_THIRD_PARTY_NOTICES@]==])
set(A2F_SDK_LICENSE [==[@A2F_SDK_LICENSE@]==])
set(A2F_CUDA_LICENSE [==[@A2F_CUDA_LICENSE@]==])
set(A2F_TENSORRT_LICENSE [==[@A2F_TENSORRT_LICENSE@]==])
set(A2F_TENSORRT_ACKNOWLEDGEMENTS [==[@A2F_TENSORRT_ACKNOWLEDGEMENTS@]==])
set(A2F_AUDIO2FACE_MODEL_LICENSE [==[@A2F_AUDIO2FACE_MODEL_LICENSE@]==])
set(A2F_AUDIO2EMOTION_MODEL_LICENSE [==[@A2F_AUDIO2EMOTION_MODEL_LICENSE@]==])
]====])
  set(A2F_STAGE_ROOT "${_a2f_runtime_root}")
  set(A2F_PLATFORM "${A2F_RUNTIME_PLATFORM}")
  set(A2F_PACKAGE_VERSION "${PROJECT_VERSION}")
  set(A2F_WORKER_FILE "$<TARGET_FILE:a2f_blender_worker>")
  set(A2F_RUNTIME_FILES "${_a2f_runtime_files}")
  set(A2F_TRTEXEC_FILE "${A2F_BUNDLE_TRTEXEC}")
  set(A2F_TRTEXEC_SOURCE_LICENSE "${A2F_BUNDLE_TRTEXEC_SOURCE_LICENSE}")
  set(A2F_TRTEXEC_PROVENANCE "${A2F_BUNDLE_TRTEXEC_PROVENANCE}")
  set(A2F_AUDIO2FACE_MODEL_DIR "${A2F_BUNDLE_AUDIO2FACE_MODEL_DIR}")
  set(A2F_AUDIO2EMOTION_MODEL_DIR "${A2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR}")
  set(A2F_PROJECT_LICENSE "${_a2f_project_license}")
  set(A2F_THIRD_PARTY_NOTICES "${_a2f_third_party_notices}")
  set(A2F_SDK_LICENSE "${_a2f_sdk_license}")
  set(A2F_CUDA_LICENSE "${_a2f_cuda_license}")
  set(A2F_TENSORRT_LICENSE "${_a2f_tensorrt_license}")
  set(A2F_TENSORRT_ACKNOWLEDGEMENTS "${_a2f_tensorrt_acknowledgements}")
  set(A2F_AUDIO2FACE_MODEL_LICENSE "${A2F_BUNDLE_AUDIO2FACE_MODEL_LICENSE}")
  set(A2F_AUDIO2EMOTION_MODEL_LICENSE "${A2F_BUNDLE_AUDIO2EMOTION_MODEL_LICENSE}")
  string(CONFIGURE "${_a2f_stage_config_content}" _a2f_stage_config_content @ONLY)
  file(GENERATE
    OUTPUT "${_a2f_stage_config}"
    CONTENT "${_a2f_stage_config_content}"
    TARGET a2f_blender_worker)

  add_custom_target(a2f_blender_runtime_stage
    COMMAND "${CMAKE_COMMAND}"
      "-DA2F_STAGE_CONFIG=${_a2f_stage_config}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageRuntime.cmake"
    DEPENDS a2f_blender_worker
    COMMENT "Staging strict ${A2F_RUNTIME_PLATFORM} Audio2Face runtime"
    VERBATIM)

  set(_a2f_runtime_archive
    "${_a2f_runtime_root}/a2f-blender-runtime-${PROJECT_VERSION}-${A2F_RUNTIME_PLATFORM}.zip")
  set(_a2f_runtime_catalog_fragment
    "${_a2f_runtime_root}/a2f-blender-runtime-${PROJECT_VERSION}-${A2F_RUNTIME_PLATFORM}.catalog-fragment.json")
  add_custom_target(a2f_blender_runtime_archive
    COMMAND "${CMAKE_COMMAND}" -E rm -f
      "${_a2f_runtime_archive}" "${_a2f_runtime_catalog_fragment}"
    COMMAND "${CMAKE_COMMAND}" -E tar cf "${_a2f_runtime_archive}"
      --format=zip "runtime/${A2F_RUNTIME_PLATFORM}"
    COMMAND "${CMAKE_COMMAND}"
      "-DA2F_ARCHIVE_FILE=${_a2f_runtime_archive}"
      "-DA2F_ARCHIVE_RUNTIME_DIRECTORY=${_a2f_runtime_root}/${A2F_RUNTIME_PLATFORM}"
      "-DA2F_ARCHIVE_FRAGMENT=${_a2f_runtime_catalog_fragment}"
      "-DA2F_ARCHIVE_PLATFORM=${A2F_RUNTIME_PLATFORM}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MeasureRuntimeArchive.cmake"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Creating downloadable Audio2Face runtime archive ${_a2f_runtime_archive}"
    VERBATIM)
  add_dependencies(a2f_blender_runtime_archive a2f_blender_runtime_stage)

  message(STATUS
    "Audio2Face runtime packaging enabled: build target "
    "a2f_blender_runtime_archive -> ${_a2f_runtime_archive} and "
    "${_a2f_runtime_catalog_fragment}")
endif()
