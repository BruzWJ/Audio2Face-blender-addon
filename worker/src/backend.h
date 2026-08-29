#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
  std::uint32_t sample_rate;
  json settings;
};

struct StreamFrame {
  std::int64_t timestamp_sample;
  std::vector<float> weights;
  std::vector<float> effective_emotions;
};

using StreamFrameCallback = std::function<void(const StreamFrame& frame)>;

struct TrackRequest {
  std::uint32_t sample_rate;
};

struct TrackRenderRequest {
  std::uint64_t revision;
  json settings;
  std::optional<std::int64_t> preview_sample;
};

using TrackPreviewCallback = std::function<void(const StreamFrame& frame)>;
using TrackCacheCallback =
    std::function<void(const std::vector<StreamFrame>& frames)>;

class Backend final {
 public:
  Backend();
  ~Backend();

  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  json load_model(const ModelRequest& request);
  json stream_start(const StreamRequest& request);
  void stream_chunk(const std::vector<float>& audio,
                    std::atomic_bool& canceled,
                    const StreamFrameCallback& frame);
  void stream_settings(const json& settings,
                       std::atomic_bool& canceled);
  void stream_end(std::atomic_bool& canceled,
                  const StreamFrameCallback& frame);
  void track_start(const TrackRequest& request);
  void track_chunk(const std::vector<float>& audio,
                   std::atomic_bool& canceled);
  void track_prepare(std::atomic_bool& canceled);
  std::size_t track_render(
      const TrackRenderRequest& request,
      std::atomic_bool& canceled,
      const std::atomic<std::uint64_t>& latest_revision,
      const TrackPreviewCallback& preview,
      const TrackCacheCallback& cache);
  void interrupt_operation() noexcept;
  void abort_operation() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace a2f_worker
