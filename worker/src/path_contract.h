#pragma once

#include <filesystem>
#include <string>

namespace a2f_worker {

std::filesystem::path require_canonical_regular_file(
    const std::string& value, const char* error_code, const char* label);

std::filesystem::path require_canonical_new_file(
    const std::string& value, const char* invalid_code,
    const char* exists_code, const char* label);

}  // namespace a2f_worker
