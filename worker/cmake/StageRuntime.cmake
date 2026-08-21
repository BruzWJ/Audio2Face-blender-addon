# Stage a downloadable Audio2Face runtime package.

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED A2F_STAGE_CONFIG OR A2F_STAGE_CONFIG STREQUAL "")
  message(FATAL_ERROR "A2F_STAGE_CONFIG must name the generated runtime staging config")
endif()
if(NOT EXISTS "${A2F_STAGE_CONFIG}")
  message(FATAL_ERROR "Runtime staging config does not exist: ${A2F_STAGE_CONFIG}")
endif()
include("${A2F_STAGE_CONFIG}")

foreach(_a2f_required IN ITEMS
    A2F_STAGE_ROOT
    A2F_PLATFORM
    A2F_PACKAGE_VERSION
    A2F_WORKER_FILE
    A2F_RUNTIME_FILES
    A2F_TRTEXEC_FILE
    A2F_TRTEXEC_SOURCE_LICENSE
    A2F_TRTEXEC_PROVENANCE
    A2F_AUDIO2FACE_MODEL_DIR
    A2F_AUDIO2EMOTION_MODEL_DIR
    A2F_PROJECT_LICENSE
    A2F_THIRD_PARTY_NOTICES
    A2F_SDK_LICENSE
    A2F_CUDA_LICENSE
    A2F_TENSORRT_LICENSE
    A2F_TENSORRT_ACKNOWLEDGEMENTS
    A2F_AUDIO2FACE_MODEL_LICENSE
    A2F_AUDIO2EMOTION_MODEL_LICENSE)
  if(NOT DEFINED ${_a2f_required} OR "${${_a2f_required}}" STREQUAL "")
    message(FATAL_ERROR "Runtime staging config is missing ${_a2f_required}")
  endif()
endforeach()

if(NOT A2F_PLATFORM STREQUAL "windows-x64" AND
   NOT A2F_PLATFORM STREQUAL "linux-x64")
  message(FATAL_ERROR "Unsupported runtime platform tag: ${A2F_PLATFORM}")
endif()

get_filename_component(_a2f_stage_root "${A2F_STAGE_ROOT}" ABSOLUTE)
set(_a2f_stage_directory "${_a2f_stage_root}/${A2F_PLATFORM}")
get_filename_component(_a2f_stage_root_name "${_a2f_stage_root}" NAME)
get_filename_component(_a2f_stage_name "${_a2f_stage_directory}" NAME)
if(NOT _a2f_stage_root_name STREQUAL "runtime" OR
   NOT _a2f_stage_name STREQUAL A2F_PLATFORM OR
   _a2f_stage_directory STREQUAL "/" OR
   _a2f_stage_directory MATCHES "^[A-Za-z]:[/\\\\]$")
  message(FATAL_ERROR "Refusing unsafe runtime stage path: ${_a2f_stage_directory}")
endif()

if(NOT A2F_PACKAGE_VERSION MATCHES
   "^[0-9]+\\.[0-9]+\\.[0-9]+([-.][A-Za-z0-9.-]+)?$")
  message(FATAL_ERROR "Invalid staged runtime version: ${A2F_PACKAGE_VERSION}")
endif()

# Invalidate same-version artifacts before checking inputs. A failed release
# staging run must not leave a stale, apparently publishable archive behind.
file(REMOVE
  "${_a2f_stage_root}/audio2face-runtime-${A2F_PACKAGE_VERSION}-${A2F_PLATFORM}.zip"
  "${_a2f_stage_root}/audio2face-runtime-${A2F_PACKAGE_VERSION}-${A2F_PLATFORM}.catalog-fragment.json")

foreach(_a2f_file IN ITEMS
    "${A2F_WORKER_FILE}"
    "${A2F_TRTEXEC_FILE}"
    "${A2F_TRTEXEC_SOURCE_LICENSE}"
    "${A2F_TRTEXEC_PROVENANCE}"
    "${A2F_PROJECT_LICENSE}"
    "${A2F_THIRD_PARTY_NOTICES}"
    "${A2F_SDK_LICENSE}"
    "${A2F_CUDA_LICENSE}"
    "${A2F_TENSORRT_LICENSE}"
    "${A2F_TENSORRT_ACKNOWLEDGEMENTS}"
    "${A2F_AUDIO2FACE_MODEL_LICENSE}"
    "${A2F_AUDIO2EMOTION_MODEL_LICENSE}")
  if(NOT EXISTS "${_a2f_file}" OR IS_DIRECTORY "${_a2f_file}")
    message(FATAL_ERROR "Required runtime package input is not a file: ${_a2f_file}")
  endif()
  file(SIZE "${_a2f_file}" _a2f_required_file_size)
  if(_a2f_required_file_size LESS 1)
    message(FATAL_ERROR "Required runtime package input is empty: ${_a2f_file}")
  endif()
endforeach()
function(a2f_validate_model_input directory label)
  if(NOT IS_DIRECTORY "${directory}")
    message(FATAL_ERROR "${label} model input is not a directory: ${directory}")
  endif()
  foreach(_a2f_model_file IN ITEMS model.json network.onnx trt_info.json)
    if(NOT EXISTS "${directory}/${_a2f_model_file}" OR
       IS_DIRECTORY "${directory}/${_a2f_model_file}")
      message(FATAL_ERROR
        "Official ${label} ONNX model input is missing ${_a2f_model_file}")
    endif()
    file(SIZE "${directory}/${_a2f_model_file}" _a2f_model_file_size)
    if(_a2f_model_file_size LESS 1)
      message(FATAL_ERROR
        "Official ${label} ONNX model input is empty: ${_a2f_model_file}")
    endif()
  endforeach()
  file(GLOB_RECURSE _a2f_model_inputs LIST_DIRECTORIES FALSE
    "${directory}/*")
  set(_a2f_input_engines)
  foreach(_a2f_model_input IN LISTS _a2f_model_inputs)
    get_filename_component(_a2f_model_extension "${_a2f_model_input}" LAST_EXT)
    string(TOLOWER "${_a2f_model_extension}" _a2f_model_extension)
    if(_a2f_model_extension STREQUAL ".trt" OR
       _a2f_model_extension STREQUAL ".engine")
      list(APPEND _a2f_input_engines "${_a2f_model_input}")
    endif()
  endforeach()
  if(_a2f_input_engines)
    message(FATAL_ERROR
      "Official ${label} model input must not contain a prebuilt TensorRT "
      "engine: ${_a2f_input_engines}")
  endif()
endfunction()

a2f_validate_model_input("${A2F_AUDIO2FACE_MODEL_DIR}"
  "Audio2Face-3D v3.0")
a2f_validate_model_input("${A2F_AUDIO2EMOTION_MODEL_DIR}"
  "Audio2Emotion v3.0")

function(a2f_validate_x64_executable path label)
  file(SIZE "${path}" _a2f_executable_size)
  if(A2F_PLATFORM STREQUAL "linux-x64")
    if(_a2f_executable_size LESS 20)
      message(FATAL_ERROR "${label} is too small to be an ELF64 executable: ${path}")
    endif()
    file(READ "${path}" _a2f_elf_header OFFSET 0 LIMIT 20 HEX)
    string(SUBSTRING "${_a2f_elf_header}" 0 14 _a2f_elf_ident)
    string(SUBSTRING "${_a2f_elf_header}" 36 4 _a2f_elf_machine)
    if(NOT _a2f_elf_ident STREQUAL "7f454c46020101" OR
       NOT _a2f_elf_machine STREQUAL "3e00")
      message(FATAL_ERROR "${label} is not Linux ELF64 x86-64: ${path}")
    endif()
  else()
    get_filename_component(_a2f_executable_extension "${path}" EXT)
    string(TOLOWER "${_a2f_executable_extension}" _a2f_executable_extension)
    if(NOT _a2f_executable_extension STREQUAL ".exe" OR
       _a2f_executable_size LESS 90)
      message(FATAL_ERROR "${label} is not a Windows x64 executable: ${path}")
    endif()
    file(READ "${path}" _a2f_dos_header OFFSET 0 LIMIT 64 HEX)
    string(SUBSTRING "${_a2f_dos_header}" 0 4 _a2f_dos_magic)
    string(SUBSTRING "${_a2f_dos_header}" 120 2 _a2f_pe_offset_0)
    string(SUBSTRING "${_a2f_dos_header}" 122 2 _a2f_pe_offset_1)
    string(SUBSTRING "${_a2f_dos_header}" 124 2 _a2f_pe_offset_2)
    string(SUBSTRING "${_a2f_dos_header}" 126 2 _a2f_pe_offset_3)
    math(EXPR _a2f_pe_offset
      "0x${_a2f_pe_offset_3}${_a2f_pe_offset_2}${_a2f_pe_offset_1}${_a2f_pe_offset_0}")
    math(EXPR _a2f_pe_required_size "${_a2f_pe_offset} + 26")
    if(NOT _a2f_dos_magic STREQUAL "4d5a" OR
       _a2f_pe_offset LESS 64 OR
       _a2f_pe_required_size GREATER _a2f_executable_size)
      message(FATAL_ERROR "${label} has an invalid PE header: ${path}")
    endif()
    file(READ "${path}" _a2f_pe_header OFFSET ${_a2f_pe_offset} LIMIT 26 HEX)
    string(SUBSTRING "${_a2f_pe_header}" 0 8 _a2f_pe_magic)
    string(SUBSTRING "${_a2f_pe_header}" 8 4 _a2f_pe_machine)
    string(SUBSTRING "${_a2f_pe_header}" 48 4 _a2f_pe_optional_magic)
    if(NOT _a2f_pe_magic STREQUAL "50450000" OR
       NOT _a2f_pe_machine STREQUAL "6486" OR
       NOT _a2f_pe_optional_magic STREQUAL "0b02")
      message(FATAL_ERROR "${label} is not PE32+ AMD64: ${path}")
    endif()
  endif()
endfunction()

a2f_validate_x64_executable("${A2F_WORKER_FILE}" "Audio2Face worker")
a2f_validate_x64_executable("${A2F_TRTEXEC_FILE}" "TensorRT trtexec")

set(_a2f_runtime_names)
set(_a2f_audio2x_name "")
foreach(_a2f_runtime IN LISTS A2F_RUNTIME_FILES)
  if(NOT EXISTS "${_a2f_runtime}" OR IS_DIRECTORY "${_a2f_runtime}")
    message(FATAL_ERROR "Required reviewed runtime library is missing: ${_a2f_runtime}")
  endif()
  file(SIZE "${_a2f_runtime}" _a2f_runtime_size)
  if(_a2f_runtime_size LESS 1)
    message(FATAL_ERROR "Required reviewed runtime library is empty: ${_a2f_runtime}")
  endif()
  get_filename_component(_a2f_runtime_name "${_a2f_runtime}" NAME)
  list(FIND _a2f_runtime_names "${_a2f_runtime_name}" _a2f_duplicate_index)
  if(NOT _a2f_duplicate_index EQUAL -1)
    message(FATAL_ERROR
      "Two reviewed runtime inputs have the same package name: ${_a2f_runtime_name}")
  endif()
  list(APPEND _a2f_runtime_names "${_a2f_runtime_name}")
  if(_a2f_runtime_name STREQUAL "audio2x.dll" OR
     _a2f_runtime_name MATCHES "^libaudio2x\\.so($|\\.)")
    set(_a2f_audio2x_name "${_a2f_runtime_name}")
  endif()
endforeach()
if(_a2f_audio2x_name STREQUAL "")
  message(FATAL_ERROR "Reviewed runtime list does not contain audio2x.dll/libaudio2x.so")
endif()

# The target is deliberately clean and entirely build-local. A failed staging
# command cannot leave an apparently complete package from a previous build.
file(REMOVE_RECURSE "${_a2f_stage_directory}")
file(MAKE_DIRECTORY
  "${_a2f_stage_directory}/bin"
  "${_a2f_stage_directory}/lib"
  "${_a2f_stage_directory}/models"
  "${_a2f_stage_directory}/licenses")

get_filename_component(_a2f_worker_name "${A2F_WORKER_FILE}" NAME)
get_filename_component(_a2f_trtexec_name "${A2F_TRTEXEC_FILE}" NAME)
if(A2F_PLATFORM STREQUAL "linux-x64")
  set(_a2f_expected_worker_name "audio2face_worker")
  set(_a2f_expected_trtexec_name "trtexec")
else()
  set(_a2f_expected_worker_name "audio2face_worker.exe")
  set(_a2f_expected_trtexec_name "trtexec.exe")
endif()
if(NOT _a2f_worker_name STREQUAL _a2f_expected_worker_name)
  message(FATAL_ERROR
    "Production worker must use package filename ${_a2f_expected_worker_name}; "
    "got ${_a2f_worker_name}")
endif()
if(NOT _a2f_trtexec_name STREQUAL _a2f_expected_trtexec_name)
  message(FATAL_ERROR
    "TensorRT executable must use package filename ${_a2f_expected_trtexec_name}; "
    "got ${_a2f_trtexec_name}")
endif()
if(_a2f_worker_name STREQUAL _a2f_trtexec_name)
  message(FATAL_ERROR
    "Worker and trtexec inputs cannot have the same package filename: "
    "${_a2f_worker_name}")
endif()

# configure_file(COPYONLY) materializes the target bytes of any reviewed input
# symlink under the explicitly selected SONAME/DLL filename. This is important:
# the secure installer rejects ZIP symlink entries, while the dynamic loader
# still needs the reviewed dependency name in lib/.
configure_file("${A2F_WORKER_FILE}"
  "${_a2f_stage_directory}/bin/${_a2f_worker_name}" COPYONLY)
configure_file("${A2F_TRTEXEC_FILE}"
  "${_a2f_stage_directory}/bin/${_a2f_trtexec_name}" COPYONLY)
foreach(_a2f_runtime IN LISTS A2F_RUNTIME_FILES)
  get_filename_component(_a2f_runtime_name "${_a2f_runtime}" NAME)
  configure_file("${_a2f_runtime}"
    "${_a2f_stage_directory}/lib/${_a2f_runtime_name}" COPYONLY)
endforeach()

# Preserve both complete reviewed v3.0 model trees. VCS/cache debris and
# GPU-specific engines are never part of the release. Any model symlink copied
# by CMake is rejected by the all-payload check below.
foreach(_a2f_model_name IN ITEMS audio2face audio2emotion)
  if(_a2f_model_name STREQUAL "audio2face")
    set(_a2f_model_source "${A2F_AUDIO2FACE_MODEL_DIR}")
  else()
    set(_a2f_model_source "${A2F_AUDIO2EMOTION_MODEL_DIR}")
  endif()
  file(COPY "${_a2f_model_source}/"
    DESTINATION "${_a2f_stage_directory}/models/${_a2f_model_name}"
    PATTERN ".git" EXCLUDE
    PATTERN ".cache" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.trt" EXCLUDE
    PATTERN "*.engine" EXCLUDE)
endforeach()

configure_file("${A2F_PROJECT_LICENSE}"
  "${_a2f_stage_directory}/licenses/audio2face-LICENSE.txt" COPYONLY)
configure_file("${A2F_THIRD_PARTY_NOTICES}"
  "${_a2f_stage_directory}/licenses/THIRD_PARTY_NOTICES.md" COPYONLY)
configure_file("${A2F_SDK_LICENSE}"
  "${_a2f_stage_directory}/licenses/audio2face-sdk-LICENSE.txt" COPYONLY)
configure_file("${A2F_CUDA_LICENSE}"
  "${_a2f_stage_directory}/licenses/cuda-LICENSE.txt" COPYONLY)
configure_file("${A2F_TENSORRT_LICENSE}"
  "${_a2f_stage_directory}/licenses/tensorrt-LICENSE.txt" COPYONLY)
configure_file("${A2F_TENSORRT_ACKNOWLEDGEMENTS}"
  "${_a2f_stage_directory}/licenses/tensorrt-ACKNOWLEDGEMENTS.txt" COPYONLY)
configure_file("${A2F_TRTEXEC_SOURCE_LICENSE}"
  "${_a2f_stage_directory}/licenses/trtexec-source-LICENSE.txt" COPYONLY)
configure_file("${A2F_TRTEXEC_PROVENANCE}"
  "${_a2f_stage_directory}/licenses/trtexec-PROVENANCE.txt" COPYONLY)
configure_file("${A2F_AUDIO2FACE_MODEL_LICENSE}"
  "${_a2f_stage_directory}/licenses/audio2face-model-LICENSE.txt" COPYONLY)
configure_file("${A2F_AUDIO2EMOTION_MODEL_LICENSE}"
  "${_a2f_stage_directory}/licenses/audio2emotion-model-LICENSE.txt" COPYONLY)

set(_a2f_staged_worker "${_a2f_stage_directory}/bin/${_a2f_worker_name}")
set(_a2f_staged_trtexec "${_a2f_stage_directory}/bin/${_a2f_trtexec_name}")
if(A2F_PLATFORM STREQUAL "linux-x64")
  file(CHMOD "${_a2f_staged_worker}" "${_a2f_staged_trtexec}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE)
endif()

foreach(_a2f_staged_model_name IN ITEMS audio2face audio2emotion)
  foreach(_a2f_staged_model_file IN ITEMS model.json network.onnx trt_info.json)
    if(NOT EXISTS
       "${_a2f_stage_directory}/models/${_a2f_staged_model_name}/${_a2f_staged_model_file}")
      message(FATAL_ERROR
        "Staged ${_a2f_staged_model_name} model is incomplete: "
        "${_a2f_staged_model_file} was not copied")
    endif()
  endforeach()
endforeach()
file(GLOB_RECURSE _a2f_staged_engines LIST_DIRECTORIES FALSE
  "${_a2f_stage_directory}/models/*.trt"
  "${_a2f_stage_directory}/models/*.engine")
if(_a2f_staged_engines)
  message(FATAL_ERROR
    "Staged runtime unexpectedly contains a prebuilt TensorRT engine: ${_a2f_staged_engines}")
endif()

# bundle.json is deliberately the exact resolver contract. Release/catalog
# metadata belongs beside the archive, not inside this launch manifest.
string(CONCAT _a2f_bundle_json
  "{\n"
  "  \"schema\": \"audio2face-runtime/2\",\n"
  "  \"platform\": \"${A2F_PLATFORM}\",\n"
  "  \"worker\": \"bin/${_a2f_worker_name}\",\n"
  "  \"trtexec\": \"bin/${_a2f_trtexec_name}\",\n"
  "  \"audio2face_model\": \"models/audio2face/model.json\",\n"
  "  \"audio2emotion_model\": \"models/audio2emotion/model.json\",\n"
  "  \"library_directories\": [\"lib\"],\n"
  "  \"licenses\": [\n"
  "    \"licenses/audio2face-LICENSE.txt\",\n"
  "    \"licenses/THIRD_PARTY_NOTICES.md\",\n"
  "    \"licenses/audio2face-sdk-LICENSE.txt\",\n"
  "    \"licenses/cuda-LICENSE.txt\",\n"
  "    \"licenses/tensorrt-LICENSE.txt\",\n"
  "    \"licenses/tensorrt-ACKNOWLEDGEMENTS.txt\",\n"
  "    \"licenses/trtexec-source-LICENSE.txt\",\n"
  "    \"licenses/trtexec-PROVENANCE.txt\",\n"
  "    \"licenses/audio2face-model-LICENSE.txt\",\n"
  "    \"licenses/audio2emotion-model-LICENSE.txt\"\n"
  "  ]\n"
  "}\n")
file(WRITE "${_a2f_stage_directory}/bundle.json" "${_a2f_bundle_json}")

set(_a2f_required_license_names
  audio2face-LICENSE.txt
  THIRD_PARTY_NOTICES.md
  audio2face-sdk-LICENSE.txt
  cuda-LICENSE.txt
  tensorrt-LICENSE.txt
  tensorrt-ACKNOWLEDGEMENTS.txt
  trtexec-source-LICENSE.txt
  trtexec-PROVENANCE.txt
  audio2face-model-LICENSE.txt
  audio2emotion-model-LICENSE.txt)
foreach(_a2f_license_name IN LISTS _a2f_required_license_names)
  if(NOT EXISTS "${_a2f_stage_directory}/licenses/${_a2f_license_name}" OR
     IS_DIRECTORY "${_a2f_stage_directory}/licenses/${_a2f_license_name}")
    message(FATAL_ERROR "Staged runtime license is missing: ${_a2f_license_name}")
  endif()
endforeach()
foreach(_a2f_runtime_name IN LISTS _a2f_runtime_names)
  if(NOT EXISTS "${_a2f_stage_directory}/lib/${_a2f_runtime_name}" OR
     IS_DIRECTORY "${_a2f_stage_directory}/lib/${_a2f_runtime_name}")
    message(FATAL_ERROR "Staged reviewed runtime is missing: ${_a2f_runtime_name}")
  endif()
endforeach()
foreach(_a2f_required_directory IN ITEMS bin lib models licenses)
  file(GLOB _a2f_directory_entries
    "${_a2f_stage_directory}/${_a2f_required_directory}/*")
  if(NOT _a2f_directory_entries)
    message(FATAL_ERROR
      "Staged runtime directory is empty: ${_a2f_required_directory}/")
  endif()
endforeach()
file(GLOB_RECURSE _a2f_staged_entries
  LIST_DIRECTORIES TRUE
  "${_a2f_stage_directory}/*")
foreach(_a2f_staged_entry IN LISTS _a2f_staged_entries)
  if(IS_SYMLINK "${_a2f_staged_entry}")
    message(FATAL_ERROR
      "Staged runtime must contain regular files, not symlinks: ${_a2f_staged_entry}")
  endif()
endforeach()

message(STATUS
  "Staged complete ${A2F_PLATFORM} Audio2Face runtime at ${_a2f_stage_directory}")
