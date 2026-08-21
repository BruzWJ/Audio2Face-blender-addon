# Stage the one package map generated from audio2face/runtime_contract.py.

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED A2F_STAGE_CONFIG OR A2F_STAGE_CONFIG STREQUAL "")
  message(FATAL_ERROR "A2F_STAGE_CONFIG must name the generated runtime staging config")
endif()
if(NOT EXISTS "${A2F_STAGE_CONFIG}" OR IS_DIRECTORY "${A2F_STAGE_CONFIG}")
  message(FATAL_ERROR "Runtime staging config is not a file: ${A2F_STAGE_CONFIG}")
endif()
include("${A2F_STAGE_CONFIG}")

foreach(_a2f_required IN ITEMS
    A2F_STAGE_ROOT
    A2F_PLATFORM
    A2F_WORKER_FILE
    A2F_AUDIO2X_FILE
    A2F_WORKER_PATH
    A2F_AUDIO2X_PATH
    A2F_TRTEXEC_PATH
    A2F_EXTERNAL_SOURCES
    A2F_EXTERNAL_PATHS)
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

set(_a2f_sources "${A2F_WORKER_FILE}" "${A2F_AUDIO2X_FILE}")
list(APPEND _a2f_sources ${A2F_EXTERNAL_SOURCES})
set(_a2f_paths "${A2F_WORKER_PATH}" "${A2F_AUDIO2X_PATH}")
list(APPEND _a2f_paths ${A2F_EXTERNAL_PATHS})
list(LENGTH _a2f_sources _a2f_source_count)
list(LENGTH _a2f_paths _a2f_path_count)
if(NOT _a2f_source_count EQUAL _a2f_path_count)
  message(FATAL_ERROR
    "Runtime staging source/path counts differ: "
    "${_a2f_source_count} sources, ${_a2f_path_count} paths")
endif()
list(FIND _a2f_paths "${A2F_TRTEXEC_PATH}" _a2f_trtexec_index)
if(_a2f_trtexec_index LESS 0)
  message(FATAL_ERROR "Runtime staging map does not contain the declared trtexec path")
endif()

set(_a2f_seen_paths)
set(_a2f_seen_sources)
math(EXPR _a2f_last_index "${_a2f_source_count} - 1")
foreach(_a2f_index RANGE 0 ${_a2f_last_index})
  list(GET _a2f_sources ${_a2f_index} _a2f_source)
  list(GET _a2f_paths ${_a2f_index} _a2f_path)
  if(NOT EXISTS "${_a2f_source}" OR IS_DIRECTORY "${_a2f_source}")
    message(FATAL_ERROR "Runtime package source is not a file: ${_a2f_source}")
  endif()
  file(SIZE "${_a2f_source}" _a2f_source_size)
  if(_a2f_source_size LESS 1)
    message(FATAL_ERROR "Runtime package source is empty: ${_a2f_source}")
  endif()
  file(REAL_PATH "${_a2f_source}" _a2f_real_source)
  list(FIND _a2f_seen_sources "${_a2f_real_source}" _a2f_source_duplicate)
  if(NOT _a2f_source_duplicate EQUAL -1)
    message(FATAL_ERROR "Runtime staging map repeats source ${_a2f_real_source}")
  endif()
  list(APPEND _a2f_seen_sources "${_a2f_real_source}")

  if(IS_ABSOLUTE "${_a2f_path}" OR
     _a2f_path MATCHES "\\\\" OR
     NOT (_a2f_path STREQUAL "bundle.json" OR
          _a2f_path MATCHES "^(bin|lib|licenses)/[^/]+$"))
    message(FATAL_ERROR "Runtime package path is unsafe: ${_a2f_path}")
  endif()
  if(A2F_PLATFORM STREQUAL "windows-x64" AND _a2f_path MATCHES "^lib/")
    message(FATAL_ERROR "Windows runtime package path cannot use lib/: ${_a2f_path}")
  endif()
  list(FIND _a2f_seen_paths "${_a2f_path}" _a2f_path_duplicate)
  if(NOT _a2f_path_duplicate EQUAL -1)
    message(FATAL_ERROR "Runtime staging map repeats package path ${_a2f_path}")
  endif()
  list(APPEND _a2f_seen_paths "${_a2f_path}")
endforeach()

# The stage directory is build-local and exact. Python validates the complete
# contract immediately after this target finishes.
file(REMOVE_RECURSE "${_a2f_stage_directory}")
foreach(_a2f_index RANGE 0 ${_a2f_last_index})
  list(GET _a2f_sources ${_a2f_index} _a2f_source)
  list(GET _a2f_paths ${_a2f_index} _a2f_path)
  get_filename_component(_a2f_parent "${_a2f_path}" DIRECTORY)
  if(NOT _a2f_parent STREQUAL "")
    file(MAKE_DIRECTORY "${_a2f_stage_directory}/${_a2f_parent}")
  endif()
  configure_file("${_a2f_source}"
    "${_a2f_stage_directory}/${_a2f_path}" COPYONLY)
endforeach()

if(A2F_PLATFORM STREQUAL "linux-x64")
  file(CHMOD
    "${_a2f_stage_directory}/${A2F_WORKER_PATH}"
    "${_a2f_stage_directory}/${A2F_TRTEXEC_PATH}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE)
endif()

message(STATUS
  "Staged contract-defined ${A2F_PLATFORM} Audio2Face runtime at "
  "${_a2f_stage_directory}")
