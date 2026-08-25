#include "path_contract.h"

#include "backend.h"

#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace a2f_worker {
namespace {

namespace fs = std::filesystem;

std::string display_path(const fs::path& path) { return path.u8string(); }

fs::path require_canonical_absolute(const std::string& value,
                                    const char* error_code,
                                    const char* label) {
  if (value.empty() || value.find('\0') != std::string::npos) {
    throw WorkerError(error_code,
                      std::string(label) + " is not a valid UTF-8 path",
                      {{"path", value}});
  }
  fs::path path;
  try {
    path = fs::u8path(value);
  } catch (const std::exception& error) {
    throw WorkerError(error_code,
                      std::string(label) + " is not a valid UTF-8 path",
                      {{"path", value}, {"error", error.what()}});
  }
  fs::path canonical = path.lexically_normal();
  canonical.make_preferred();
  if (!path.is_absolute() || path.filename().empty() ||
      value != canonical.u8string()) {
    throw WorkerError(
        error_code,
        std::string(label) + " must be one canonical absolute file path",
        {{"path", value}});
  }
  return path;
}

bool is_reparse_point(const fs::path& path, std::error_code& error) {
#ifdef _WIN32
  const DWORD attributes = GetFileAttributesW(path.c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    error = std::error_code(static_cast<int>(GetLastError()),
                            std::system_category());
    return false;
  }
  error.clear();
  return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#else
  (void)path;
  error.clear();
  return false;
#endif
}

void require_unaliased_components(const fs::path& path,
                                  const char* error_code,
                                  const char* label) {
  for (fs::path component = path;; component = component.parent_path()) {
    std::error_code status_error;
    const fs::file_status status = fs::symlink_status(component, status_error);
    std::error_code reparse_error;
    const bool reparse = is_reparse_point(component, reparse_error);
    if (status_error || reparse_error) {
      const std::error_code& error = status_error ? status_error : reparse_error;
      throw WorkerError(
          error_code, std::string(label) + " is missing or inaccessible",
          {{"path", display_path(path)},
           {"component", display_path(component)},
           {"error", error.message()}});
    }
    if (fs::is_symlink(status) || reparse) {
      throw WorkerError(
          error_code, std::string(label) + " must not use a filesystem alias",
          {{"path", display_path(path)},
           {"component", display_path(component)}});
    }
    if (component == component.root_path()) return;
    const fs::path parent = component.parent_path();
    if (parent.empty() || parent == component) {
      throw WorkerError(
          error_code, std::string(label) + " has no canonical filesystem root",
          {{"path", display_path(path)}});
    }
  }
}

}  // namespace

void require_canonical_regular_file(const std::string& value,
                                    const char* error_code,
                                    const char* label) {
  const fs::path path = require_canonical_absolute(value, error_code, label);
  require_unaliased_components(path, error_code, label);
  std::error_code error;
  if (!fs::is_regular_file(path, error) || error) {
    throw WorkerError(
        error_code, std::string(label) + " must be a regular file",
        {{"path", display_path(path)}, {"error", error.message()}});
  }
}

}  // namespace a2f_worker
