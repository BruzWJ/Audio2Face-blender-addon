#include "result_file.h"

#include "backend.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace a2f_worker {
namespace {

std::atomic_uint64_t g_temp_counter{0};

std::string temp_suffix() {
#ifdef _WIN32
  const auto process = static_cast<unsigned long>(GetCurrentProcessId());
#else
  const auto process = static_cast<unsigned long>(getpid());
#endif
  std::random_device random;
  std::ostringstream suffix;
  suffix << ".tmp." << process << '.'
         << g_temp_counter.fetch_add(1, std::memory_order_relaxed) << '.'
         << std::hex << std::setw(8) << std::setfill('0') << random()
         << std::setw(8) << random();
  return suffix.str();
}

void atomic_publish(const std::filesystem::path& temporary,
                    const std::filesystem::path& target) {
#ifdef _WIN32
  if (!MoveFileW(temporary.c_str(), target.c_str())) {
    const auto win32_error = GetLastError();
    const char* code = (win32_error == ERROR_ALREADY_EXISTS ||
                        win32_error == ERROR_FILE_EXISTS)
                           ? "result_exists"
                           : "result_commit_failed";
    throw WorkerError(code, "Could not atomically commit result file",
                      {{"path", target.string()}, {"win32_error", win32_error}});
  }
#else
  // Both paths are in the same directory. link() publishes the complete file
  // only if the result target is still absent.
  if (::link(temporary.c_str(), target.c_str()) != 0) {
    const int error = errno;
    throw WorkerError(error == EEXIST ? "result_exists" : "result_commit_failed",
                      "Could not atomically commit result file",
                      {{"path", target.string()}, {"errno", error},
                       {"error", std::strerror(error)}});
  }
  (void)::unlink(temporary.c_str());
#endif
}

}  // namespace

void write_json_atomically(
    const std::string& path, const nlohmann::json& value,
    const std::atomic_bool& canceled,
    const ResultPublicationGate& publication_gate) {
  namespace fs = std::filesystem;
  const fs::path target(path);
  if (!target.is_absolute() || target.filename().empty()) {
    throw WorkerError("invalid_result_path", "Result path must be an absolute file path",
                      {{"path", path}});
  }
  std::error_code error;
  if (fs::exists(target, error)) {
    throw WorkerError("result_exists", "Result file already exists", {{"path", target.string()}});
  }
  if (error) {
    throw WorkerError("invalid_result_path", "Could not inspect result path",
                      {{"path", target.string()}, {"error", error.message()}});
  }
  const fs::path directory = target.parent_path();
  if (!fs::is_directory(directory, error) || error) {
    throw WorkerError("result_directory_failed", "Result directory does not exist",
                      {{"path", directory.string()}, {"error", error.message()}});
  }

  const fs::path temporary(target.string() + temp_suffix());
  try {
    if (canceled.load(std::memory_order_relaxed)) {
      throw WorkerError("canceled", "Generation was canceled before result publication");
    }
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw WorkerError("result_write_failed", "Could not open temporary result file",
                        {{"path", temporary.string()}});
    }
    // Stream the serializer directly so a large animation document is not
    // duplicated into one temporary std::string before the file write.
    stream << value;
    stream.flush();
    if (!stream) {
      throw WorkerError("result_write_failed", "Could not write temporary result file",
                        {{"path", temporary.string()}});
    }
    stream.close();
    if (canceled.load(std::memory_order_relaxed)) {
      throw WorkerError("canceled", "Generation was canceled during result serialization");
    }
    publication_gate(
        [&temporary, &target] { atomic_publish(temporary, target); });
  } catch (...) {
    std::error_code cleanup_error;
    fs::remove(temporary, cleanup_error);
    throw;
  }
}

}  // namespace a2f_worker
