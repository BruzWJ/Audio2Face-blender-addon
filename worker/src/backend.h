#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace a2f_worker {

using json = nlohmann::json;

class WorkerError final : public std::runtime_error {
 public:
  WorkerError(std::string code, std::string message, json details = json::object());

  const std::string& code() const noexcept { return code_; }
  const json& details() const noexcept { return details_; }

 private:
  std::string code_;
  json details_;
};

struct ModelRequest {
  std::string audio2face_model_path;
  std::string audio2emotion_model_path;
};

struct StreamRequest {
  std::string operation_id;
  std::uint32_t sample_rate;
  json settings;
};

struct StreamFrame {
  std::int64_t timestamp_sample;
  std::vector<float> weights;
};

using StreamFrameCallback = std::function<void(const StreamFrame& frame)>;

class Backend final {
 public:
  Backend();
  ~Backend();

  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  json load_model(const ModelRequest& request);
  json stream_start(const StreamRequest& request);
  void stream_chunk(const std::string& operation_id,
                    const std::vector<float>& audio,
                    std::atomic_bool& canceled,
                    const StreamFrameCallback& frame);
  void stream_end(const std::string& operation_id,
                  std::atomic_bool& canceled,
                  const StreamFrameCallback& frame);
  void stream_abort(const std::string& operation_id) noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace a2f_worker
