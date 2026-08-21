# Validate and measure a staged Audio2Face runtime archive.

cmake_minimum_required(VERSION 3.24)

foreach(_a2f_measurement_input IN ITEMS
    A2F_ARCHIVE_FILE
    A2F_ARCHIVE_RUNTIME_DIRECTORY
    A2F_ARCHIVE_FRAGMENT
    A2F_ARCHIVE_PLATFORM)
  if(NOT DEFINED ${_a2f_measurement_input} OR
     "${${_a2f_measurement_input}}" STREQUAL "")
    message(FATAL_ERROR "Archive measurement is missing ${_a2f_measurement_input}")
  endif()
endforeach()
if(NOT EXISTS "${A2F_ARCHIVE_FILE}" OR IS_DIRECTORY "${A2F_ARCHIVE_FILE}")
  message(FATAL_ERROR "Runtime archive was not created: ${A2F_ARCHIVE_FILE}")
endif()
if(NOT IS_DIRECTORY "${A2F_ARCHIVE_RUNTIME_DIRECTORY}")
  message(FATAL_ERROR
    "Staged runtime disappeared before measurement: ${A2F_ARCHIVE_RUNTIME_DIRECTORY}")
endif()
if(NOT A2F_ARCHIVE_PLATFORM STREQUAL "windows-x64" AND
   NOT A2F_ARCHIVE_PLATFORM STREQUAL "linux-x64")
  message(FATAL_ERROR "Cannot measure unsupported platform ${A2F_ARCHIVE_PLATFORM}")
endif()
get_filename_component(_a2f_archive_platform_name
  "${A2F_ARCHIVE_RUNTIME_DIRECTORY}" NAME)
get_filename_component(_a2f_archive_runtime_root
  "${A2F_ARCHIVE_RUNTIME_DIRECTORY}" DIRECTORY)
get_filename_component(_a2f_archive_runtime_name
  "${_a2f_archive_runtime_root}" NAME)
get_filename_component(_a2f_archive_package_root
  "${_a2f_archive_runtime_root}" DIRECTORY)
if(NOT _a2f_archive_platform_name STREQUAL A2F_ARCHIVE_PLATFORM OR
   NOT _a2f_archive_runtime_name STREQUAL "runtime")
  message(FATAL_ERROR
    "Archive payload must be staged as runtime/${A2F_ARCHIVE_PLATFORM}: "
    "${A2F_ARCHIVE_RUNTIME_DIRECTORY}")
endif()

file(GLOB_RECURSE _a2f_archive_payload
  LIST_DIRECTORIES FALSE
  RELATIVE "${_a2f_archive_package_root}"
  "${A2F_ARCHIVE_RUNTIME_DIRECTORY}/*")
list(SORT _a2f_archive_payload)
set(_a2f_staged_unpacked_size 0)
foreach(_a2f_payload_relative IN LISTS _a2f_archive_payload)
  set(_a2f_payload
    "${_a2f_archive_package_root}/${_a2f_payload_relative}")
  if(_a2f_payload_relative MATCHES "[;\r\n]" OR
     _a2f_payload_relative MATCHES "\\\\")
    message(FATAL_ERROR
      "Runtime archive path is not portable: ${_a2f_payload_relative}")
  endif()
  if(IS_SYMLINK "${_a2f_payload}")
    message(FATAL_ERROR "Runtime archives may not contain symlinks: ${_a2f_payload}")
  endif()
  file(SIZE "${_a2f_payload}" _a2f_payload_size)
  math(EXPR _a2f_staged_unpacked_size
    "${_a2f_staged_unpacked_size} + ${_a2f_payload_size}")
endforeach()
if(NOT _a2f_archive_payload OR _a2f_staged_unpacked_size LESS 1)
  message(FATAL_ERROR "Staged runtime payload is empty")
endif()

# Do not publish measurements for a partial/empty archive. Compare the ZIP
# member set and member sizes to the exact regular-file staging tree first.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar tf "${A2F_ARCHIVE_FILE}"
  RESULT_VARIABLE _a2f_archive_list_result
  OUTPUT_VARIABLE _a2f_archive_member_text
  ERROR_VARIABLE _a2f_archive_list_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar tvf "${A2F_ARCHIVE_FILE}"
  RESULT_VARIABLE _a2f_archive_verbose_result
  OUTPUT_VARIABLE _a2f_archive_verbose_text
  ERROR_VARIABLE _a2f_archive_verbose_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _a2f_archive_list_result EQUAL 0 OR
   NOT _a2f_archive_verbose_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect generated runtime archive: ${_a2f_archive_list_error} "
    "${_a2f_archive_verbose_error}")
endif()
string(REPLACE "\r\n" "\n" _a2f_archive_member_text
  "${_a2f_archive_member_text}")
string(REPLACE "\r" "\n" _a2f_archive_member_text
  "${_a2f_archive_member_text}")
string(REPLACE "\r\n" "\n" _a2f_archive_verbose_text
  "${_a2f_archive_verbose_text}")
string(REPLACE "\r" "\n" _a2f_archive_verbose_text
  "${_a2f_archive_verbose_text}")
if(_a2f_archive_member_text STREQUAL "" OR
   _a2f_archive_verbose_text STREQUAL "")
  message(FATAL_ERROR "Generated runtime archive has no members")
endif()
string(REPLACE "\n" ";" _a2f_archive_members
  "${_a2f_archive_member_text}")
string(REPLACE "\n" ";" _a2f_archive_verbose_lines
  "${_a2f_archive_verbose_text}")
list(LENGTH _a2f_archive_members _a2f_archive_member_count)
list(LENGTH _a2f_archive_verbose_lines _a2f_archive_verbose_count)
if(NOT _a2f_archive_member_count EQUAL _a2f_archive_verbose_count)
  message(FATAL_ERROR "Generated runtime archive listings disagree")
endif()

set(_a2f_archive_file_members)
set(_a2f_archive_seen_members)
set(_a2f_archive_prefix "runtime/${A2F_ARCHIVE_PLATFORM}/")
set(_a2f_archive_unpacked_size 0)
math(EXPR _a2f_archive_last_member "${_a2f_archive_member_count} - 1")
foreach(_a2f_archive_index RANGE 0 ${_a2f_archive_last_member})
  list(GET _a2f_archive_members ${_a2f_archive_index} _a2f_archive_member)
  list(GET _a2f_archive_verbose_lines ${_a2f_archive_index}
    _a2f_archive_verbose_line)
  list(FIND _a2f_archive_seen_members "${_a2f_archive_member}"
    _a2f_archive_duplicate_index)
  if(NOT _a2f_archive_duplicate_index EQUAL -1)
    message(FATAL_ERROR
      "Generated runtime archive contains duplicate path: ${_a2f_archive_member}")
  endif()
  list(APPEND _a2f_archive_seen_members "${_a2f_archive_member}")
  string(FIND "${_a2f_archive_member}" "${_a2f_archive_prefix}"
    _a2f_archive_prefix_index)
  if(NOT _a2f_archive_prefix_index EQUAL 0 AND
     NOT _a2f_archive_member STREQUAL
       "runtime/${A2F_ARCHIVE_PLATFORM}")
    message(FATAL_ERROR
      "Generated archive member is outside runtime/${A2F_ARCHIVE_PLATFORM}: "
      "${_a2f_archive_member}")
  endif()
  if(_a2f_archive_member MATCHES "/$")
    continue()
  endif()
  list(APPEND _a2f_archive_file_members "${_a2f_archive_member}")
  if(NOT _a2f_archive_verbose_line MATCHES
     "^[^ ]+ +[0-9]+ +[^ ]+ +[^ ]+ +([0-9]+) +")
    message(FATAL_ERROR
      "Cannot measure generated archive member: ${_a2f_archive_verbose_line}")
  endif()
  set(_a2f_archive_member_size "${CMAKE_MATCH_1}")
  math(EXPR _a2f_archive_unpacked_size
    "${_a2f_archive_unpacked_size} + ${_a2f_archive_member_size}")
endforeach()
list(SORT _a2f_archive_file_members)
if(NOT "${_a2f_archive_file_members}" STREQUAL "${_a2f_archive_payload}")
  message(FATAL_ERROR
    "Generated runtime archive member set does not match the staged payload")
endif()
if(NOT _a2f_archive_unpacked_size EQUAL _a2f_staged_unpacked_size)
  message(FATAL_ERROR
    "Generated runtime archive sizes do not match the staged payload: "
    "${_a2f_archive_unpacked_size} vs ${_a2f_staged_unpacked_size}")
endif()

file(SHA256 "${A2F_ARCHIVE_FILE}" _a2f_archive_sha256)
file(SIZE "${A2F_ARCHIVE_FILE}" _a2f_archive_size)
if(_a2f_archive_size LESS 1 OR _a2f_archive_unpacked_size LESS 1)
  message(FATAL_ERROR "Measured runtime archive is empty")
endif()

file(WRITE "${A2F_ARCHIVE_FRAGMENT}"
  "{\n"
  "  \"${A2F_ARCHIVE_PLATFORM}\": {\n"
  "    \"sha256\": \"${_a2f_archive_sha256}\",\n"
  "    \"size\": ${_a2f_archive_size},\n"
  "    \"unpacked_size\": ${_a2f_archive_unpacked_size}\n"
  "  }\n"
  "}\n")
message(STATUS
  "Measured runtime archive: ${A2F_ARCHIVE_FRAGMENT} "
  "(${_a2f_archive_size} compressed bytes, "
  "${_a2f_archive_unpacked_size} unpacked bytes)")
