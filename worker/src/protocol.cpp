#include "protocol.h"

#include "backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace a2f_worker {

WorkerError::WorkerError(std::string code, std::string message, json details)
    : std::runtime_error(std::move(message)),
      code_(std::move(code)),
      details_(std::move(details)) {}

namespace {

constexpr const char* kProtocol = "audio2face/13";
constexpr std::size_t kMaximumRequestBytes = 1024U * 1024U;
constexpr std::size_t kStreamQueueSeconds = 4;
constexpr std::size_t kMaximumTrackChunkSamples = 64U * 1024U;
constexpr std::size_t kMaximumTrackFramesPerBatch = 64;

class Emitter {
 public:
  void response(const json& id, json result) {
    line({{"protocol", kProtocol},
          {"type", "response"},
          {"id", id},
          {"result", std::move(result)}});
  }

  void error(const std::optional<json>& id, const std::string& code,
             const std::string& message,
             const json& details = json::object()) {
    json error_value =
        {{"code", code}, {"message", message}, {"details", details}};
    json envelope = {{"protocol", kProtocol},
                     {"type", "error"},
                     {"error", std::move(error_value)}};
    if (id.has_value()) envelope["id"] = *id;
    line(envelope);
  }

  void event(const std::string& event_name, json data,
             const std::string& operation_id) {
    line({{"protocol", kProtocol},
          {"type", "event"},
          {"event", event_name},
          {"operation_id", operation_id},
          {"data", std::move(data)}});
  }

 private:
  void line(const json& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << value.dump() << '\n' << std::flush;
  }

  std::mutex mutex_;
};

int nonnegative_int(const json& value, const char* name) {
  if (value.is_number_unsigned()) {
    const auto parsed = value.get<std::uint64_t>();
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw WorkerError("invalid_params",
                        std::string(name) + " is out of range");
    }
    return static_cast<int>(parsed);
  }
  if (!value.is_number_integer()) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be a non-negative integer");
  }
  const auto parsed = value.get<std::int64_t>();
  if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw WorkerError("invalid_params", std::string(name) + " is out of range");
  }
  return static_cast<int>(parsed);
}

std::uint32_t positive_sample_rate(const json& value) {
  const int parsed = nonnegative_int(value, "sample_rate");
  if (parsed == 0) {
    throw WorkerError("invalid_params", "sample_rate must be positive");
  }
  return static_cast<std::uint32_t>(parsed);
}

std::int64_t required_nonnegative_int64(const json& value,
                                        const char* name) {
  if (!value.is_number_integer()) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be a JSON integer");
  }
  if (value.is_number_unsigned()) {
    const std::uint64_t parsed = value.get<std::uint64_t>();
    if (parsed >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw WorkerError("invalid_params", std::string(name) + " is out of range");
    }
    return static_cast<std::int64_t>(parsed);
  }
  const std::int64_t parsed = value.get<std::int64_t>();
  if (parsed < 0) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be non-negative");
  }
  return parsed;
}

std::uint64_t required_nonnegative_uint64(const json& value,
                                          const char* name) {
  if (!value.is_number_integer()) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be a JSON integer");
  }
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  const std::int64_t parsed = value.get<std::int64_t>();
  if (parsed < 0) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be non-negative");
  }
  return static_cast<std::uint64_t>(parsed);
}

std::uint64_t required_positive_revision(const json& value) {
  const std::int64_t revision =
      required_nonnegative_int64(value, "revision");
  if (revision == 0) {
    throw WorkerError("invalid_params", "revision must be positive");
  }
  return static_cast<std::uint64_t>(revision);
}

std::string required_string(const json& object, const char* name) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_string() ||
      it->get_ref<const std::string&>().empty()) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be a non-empty string");
  }
  return it->get<std::string>();
}

std::string required_operation_id(const json& params) {
  const std::string value = required_string(params, "operation_id");
  if (value.size() > 128) {
    throw WorkerError("invalid_params", "operation_id is too long");
  }
  return value;
}

void require_exact_keys(
    const json& object, std::initializer_list<const char*> expected,
    const char* code = "invalid_params",
    const char* message = "Request params do not match the method contract") {
  bool valid = object.size() == expected.size();
  json names = json::array();
  for (const char* name : expected) {
    names.push_back(name);
    valid = valid && object.contains(name);
  }
  if (!valid) {
    throw WorkerError(code, message, {{"expected", std::move(names)}});
  }
}

void apply_object_patch(json& target, const json& patch) {
  for (auto item = patch.begin(); item != patch.end(); ++item) {
    const std::string& name = item.key();
    const json& value = item.value();
    if (value.is_object() && target.contains(name) &&
        target.at(name).is_object()) {
      apply_object_patch(target[name], value);
    } else {
      target[name] = value;
    }
  }
}

std::vector<TrackSettingsEntry> parse_settings_timeline(const json& value) {
  if (!value.is_array() || value.empty()) {
    throw WorkerError("invalid_params",
                      "settings_timeline must be a non-empty array");
  }
  std::vector<TrackSettingsEntry> timeline;
  timeline.reserve(value.size());
  json settings;
  std::uint64_t previous_sample = 0;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const json& entry = value.at(index);
    if (!entry.is_object()) {
      throw WorkerError("invalid_params",
                        "settings_timeline entries must be objects",
                        {{"index", index}});
    }
    require_exact_keys(entry, {"sample", "settings"});
    const std::string sample_name =
        "settings_timeline[" + std::to_string(index) + "].sample";
    const std::uint64_t sample =
        required_nonnegative_uint64(entry.at("sample"), sample_name.c_str());
    if ((index == 0 && sample != 0) ||
        (index != 0 && sample <= previous_sample)) {
      throw WorkerError(
          "invalid_params",
          index == 0
              ? "settings_timeline must start at sample 0"
              : "settings_timeline samples must be strictly increasing",
          {{"index", index}, {"sample", sample}});
    }
    const json& patch = entry.at("settings");
    if (!patch.is_object()) {
      throw WorkerError("invalid_params",
                        "settings_timeline settings must be objects",
                        {{"index", index}});
    }
    if (index == 0) {
      settings = patch;
    } else {
      apply_object_patch(settings, patch);
    }
    timeline.push_back(TrackSettingsEntry{sample, settings});
    previous_sample = sample;
  }
  return timeline;
}

json parse_request(const std::string& line) {
  std::vector<std::set<std::string>> object_keys;
  const auto reject_duplicate_keys =
      [&object_keys](int, json::parse_event_t event, json& parsed) {
        if (event == json::parse_event_t::object_start) {
          object_keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
          const std::string key = parsed.get<std::string>();
          if (object_keys.empty() ||
              !object_keys.back().insert(key).second) {
            throw WorkerError("invalid_json",
                              "JSON object contains a duplicate key",
                              {{"key", key}});
          }
        } else if (event == json::parse_event_t::object_end) {
          object_keys.pop_back();
        }
        return true;
      };
  return json::parse(line, reject_duplicate_keys);
}

int base64_value(char value) noexcept {
  if (value >= 'A' && value <= 'Z') return value - 'A';
  if (value >= 'a' && value <= 'z') return value - 'a' + 26;
  if (value >= '0' && value <= '9') return value - '0' + 52;
  if (value == '+') return 62;
  if (value == '/') return 63;
  return -1;
}

std::vector<float> decode_f32le_base64(const json& value,
                                       std::size_t maximum_samples) {
  static_assert(sizeof(float) == sizeof(std::uint32_t),
                "Streaming protocol requires 32-bit float");
  static_assert(std::numeric_limits<float>::is_iec559,
                "Streaming protocol requires IEEE-754 float");
  if (!value.is_string()) {
    throw WorkerError("invalid_params",
                      "audio_f32le_base64 must be a base64 string");
  }
  const std::string& encoded = value.get_ref<const std::string&>();
  if (encoded.empty() || encoded.size() % 4 != 0) {
    throw WorkerError("invalid_params",
                      "audio_f32le_base64 is not canonical base64");
  }
  const std::size_t padding =
      (encoded.back() == '=') +
      (encoded.size() >= 2 && encoded[encoded.size() - 2] == '=');
  const std::size_t decoded_size = encoded.size() / 4 * 3 - padding;
  if (decoded_size == 0 || decoded_size % sizeof(float) != 0 ||
      decoded_size / sizeof(float) > maximum_samples) {
    throw WorkerError(
        "invalid_params",
        "Streaming PCM chunk must contain one non-empty bounded mono float32 block",
        {{"maximum_samples", maximum_samples}});
  }

  std::vector<std::uint8_t> bytes;
  bytes.reserve(decoded_size);
  for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
    const bool last = offset + 4 == encoded.size();
    const char c0 = encoded[offset];
    const char c1 = encoded[offset + 1];
    const char c2 = encoded[offset + 2];
    const char c3 = encoded[offset + 3];
    const int v0 = base64_value(c0);
    const int v1 = base64_value(c1);
    const int v2 = c2 == '=' ? 0 : base64_value(c2);
    const int v3 = c3 == '=' ? 0 : base64_value(c3);
    if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0 ||
        (!last && (c2 == '=' || c3 == '=')) ||
        (c2 == '=' && c3 != '=') ||
        (c2 == '=' && (v1 & 0x0f) != 0) ||
        (c3 == '=' && c2 != '=' && (v2 & 0x03) != 0)) {
      throw WorkerError("invalid_params",
                        "audio_f32le_base64 is not canonical base64");
    }
    const std::uint32_t packed =
        (static_cast<std::uint32_t>(v0) << 18U) |
        (static_cast<std::uint32_t>(v1) << 12U) |
        (static_cast<std::uint32_t>(v2) << 6U) |
        static_cast<std::uint32_t>(v3);
    bytes.push_back(static_cast<std::uint8_t>((packed >> 16U) & 0xffU));
    if (c2 != '=') {
      bytes.push_back(static_cast<std::uint8_t>((packed >> 8U) & 0xffU));
    }
    if (c3 != '=') {
      bytes.push_back(static_cast<std::uint8_t>(packed & 0xffU));
    }
  }
  if (bytes.size() != decoded_size) {
    throw WorkerError("invalid_params",
                      "audio_f32le_base64 has an invalid decoded length");
  }

  std::vector<float> samples(bytes.size() / sizeof(float));
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const std::size_t byte = index * sizeof(float);
    const std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[byte]) |
        (static_cast<std::uint32_t>(bytes[byte + 1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[byte + 2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[byte + 3]) << 24U);
    std::memcpy(&samples[index], &bits, sizeof(float));
    if (!std::isfinite(samples[index])) {
      throw WorkerError("invalid_params", "Streaming PCM must be finite",
                        {{"sample_index", index}});
    }
  }
  return samples;
}

struct StreamCommand {
  enum class Kind { Chunk, Settings, End };
  Kind kind;
  json request_id;
  std::vector<float> audio;
  json settings;
  std::future<void> response_gate;
};

struct TrackCommand {
  enum class Kind { Chunk, Prepare, Render };
  Kind kind;
  json request_id;
  std::vector<float> audio;
  std::uint64_t revision{0};
  std::vector<TrackSettingsEntry> settings_timeline;
  std::optional<std::int64_t> preview_sample;
};

enum class OperationKind { None, Stream, Track };

enum class TrackPhase { None, Uploading, Preparing, Prepared };

json track_render_response(std::uint64_t revision, std::size_t frame_count,
                           bool superseded) {
  return {{"revision", revision},
          {"frame_count", frame_count},
          {"superseded", superseded}};
}

class Server {
 public:
  ~Server() { stop_operation(); }

  int run() {
    std::string line;
    while (!shutting_down_) {
      if (!std::getline(std::cin, line)) {
        return std::cin.eof() ? 0 : 1;
      }
      if (std::cin.eof()) {
        emitter_.error(std::nullopt, "invalid_json",
                       "JSONL request must end with LF");
        return 1;
      }
      if (line.find('\r') != std::string::npos) {
        emitter_.error(std::nullopt, "invalid_json",
                       "JSONL request must not contain CR");
        continue;
      }
      if (line.size() > kMaximumRequestBytes) {
        emitter_.error(std::nullopt, "request_too_large",
                       "JSONL request exceeds 1 MiB");
        continue;
      }
      process(line);
    }
    return 0;
  }

 private:
  void process(const std::string& line) {
    std::optional<json> id;
    try {
      const json request = parse_request(line);
      if (!request.is_object()) {
        throw WorkerError("invalid_request", "Request must be an object");
      }
      if (const auto id_it = request.find("id");
          id_it != request.end() && id_it->is_string() &&
          !id_it->get_ref<const std::string&>().empty() &&
          id_it->get_ref<const std::string&>().size() <= 128) {
        id = *id_it;
      }
      require_exact_keys(
          request, {"protocol", "type", "id", "method", "params"},
          "invalid_request",
          "Request envelope does not match the protocol contract");
      if (!request.at("protocol").is_string() ||
          request.at("protocol").get_ref<const std::string&>() != kProtocol) {
        throw WorkerError("protocol_mismatch", "Unsupported protocol",
                          {{"expected", kProtocol}});
      }
      if (!request.at("type").is_string() ||
          request.at("type").get_ref<const std::string&>() != "request") {
        throw WorkerError("invalid_request", "type must be request");
      }
      if (!id.has_value()) {
        throw WorkerError(
            "invalid_request",
            "id must be a non-empty string of at most 128 characters");
      }
      if (!request.at("method").is_string() ||
          request.at("method").get_ref<const std::string&>().empty()) {
        throw WorkerError("invalid_request",
                          "method must be a non-empty string");
      }
      if (!request.at("params").is_object()) {
        throw WorkerError("invalid_params", "params must be an object");
      }
      dispatch(*id, request.at("method").get<std::string>(),
               request.at("params"));
    } catch (const WorkerError& error) {
      emitter_.error(id, error.code(), error.what(), error.details());
    } catch (const json::exception& error) {
      emitter_.error(id, "invalid_json", error.what());
    } catch (const std::exception& error) {
      emitter_.error(id, "internal_error", error.what());
    }
  }

  void dispatch(const json& id, const std::string& method,
                const json& params) {
    if (method == "hello") {
      require_exact_keys(params, {});
      emitter_.response(
          id, {{"worker_profile", "nvidia-a2f3-a2e3-gpu-arkit52/13"},
               {"worker_version", A2F_WORKER_VERSION}});
      negotiated_ = true;
      return;
    }
    if (method == "load_model") {
      require_negotiated();
      join_or_reject_active("Cannot load a model while an operation is running");
      require_exact_keys(
          params, {"audio2face_model_path", "audio2emotion_model_path"});
      emitter_.response(
          id, backend_.load_model(
                  ModelRequest{required_string(params,
                                               "audio2face_model_path"),
                               required_string(params,
                                               "audio2emotion_model_path")}));
      return;
    }
    if (method == "stream_start") {
      require_negotiated();
      start_stream(id, params);
      return;
    }
    if (method == "stream_chunk") {
      require_negotiated();
      enqueue_stream_chunk(id, params);
      return;
    }
    if (method == "stream_settings") {
      require_negotiated();
      enqueue_stream_settings(id, params);
      return;
    }
    if (method == "stream_end") {
      require_negotiated();
      enqueue_stream_end(id, params);
      return;
    }
    if (method == "track_start") {
      require_negotiated();
      start_track(id, params);
      return;
    }
    if (method == "track_chunk") {
      require_negotiated();
      enqueue_track_chunk(id, params);
      return;
    }
    if (method == "track_prepare") {
      require_negotiated();
      enqueue_track_prepare(id, params);
      return;
    }
    if (method == "track_render") {
      require_negotiated();
      enqueue_track_render(id, params);
      return;
    }
    if (method == "cancel") {
      require_negotiated();
      cancel_operation(id, params);
      return;
    }
    if (method == "shutdown") {
      require_exact_keys(params, {});
      stop_operation();
      emitter_.response(id, json::object());
      shutting_down_ = true;
      return;
    }
    throw WorkerError("method_not_found", "Unknown method",
                      {{"method", method}});
  }

  void start_stream(const json& request_id, const json& params) {
    join_or_reject_active("An operation is already running");
    require_exact_keys(params,
                       {"operation_id", "sample_rate", "settings"});
    if (!params.at("settings").is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    const std::string operation_id = required_operation_id(params);
    StreamRequest request{positive_sample_rate(params.at("sample_rate")),
                          params.at("settings")};
    std::promise<void> start_gate;
    std::future<void> start_signal = start_gate.get_future();
    std::optional<std::string> state_operation_id(operation_id);
    json response = backend_.stream_start(request);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_operation_id_ = std::move(state_operation_id);
      operation_kind_ = OperationKind::Stream;
      stream_sample_rate_ = request.sample_rate;
    }
    canceled_.store(false, std::memory_order_release);
    try {
      operation_thread_ = std::thread(
          [this, operation_id,
           start_signal = std::move(start_signal)]() mutable {
            stream_loop(operation_id, std::move(start_signal));
          });
    } catch (...) {
      backend_.abort_operation();
      finish_active();
      throw;
    }
    try {
      emitter_.response(request_id, std::move(response));
      start_gate.set_value();
    } catch (...) {
      start_gate.set_exception(std::current_exception());
      operation_thread_.join();
      throw;
    }
  }

  void enqueue_stream_chunk(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id", "audio_f32le_base64"});
    const std::string requested_operation_id = required_operation_id(params);
    std::uint32_t sample_rate = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(requested_operation_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
      sample_rate = stream_sample_rate_;
    }
    std::vector<float> audio =
        decode_f32le_base64(params.at("audio_f32le_base64"), sample_rate);
    std::promise<void> response_gate;
    StreamCommand command{StreamCommand::Kind::Chunk, {}, std::move(audio), {},
                          response_gate.get_future()};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(requested_operation_id);
      const std::size_t maximum_queued =
          static_cast<std::size_t>(stream_sample_rate_) * kStreamQueueSeconds;
      if (command.audio.size() > maximum_queued - stream_queued_samples_) {
        throw WorkerError("backpressure",
                          "Streaming PCM queue is full",
                          {{"maximum_queued_samples", maximum_queued}});
      }
      const std::size_t command_samples = command.audio.size();
      stream_queue_.push_back(std::move(command));
      stream_queued_samples_ += command_samples;
    }
    operation_condition_.notify_one();
    respond_and_release(request_id, response_gate);
  }

  void enqueue_stream_settings(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id", "settings"});
    const std::string operation_id = required_operation_id(params);
    if (!params.at("settings").is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(operation_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
      stream_queue_.push_back(StreamCommand{
          StreamCommand::Kind::Settings,
          request_id,
          {},
          params.at("settings"),
          {},
      });
      pending_operation_request_ids_.insert(request_id.get<std::string>());
    }
    operation_condition_.notify_one();
  }

  void enqueue_stream_end(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id"});
    const std::string requested_operation_id = required_operation_id(params);
    std::promise<void> response_gate;
    StreamCommand command{StreamCommand::Kind::End, {}, {}, {},
                          response_gate.get_future()};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(requested_operation_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
      stream_queue_.push_back(std::move(command));
      stream_end_queued_ = true;
    }
    operation_condition_.notify_one();
    respond_and_release(request_id, response_gate);
  }

  void start_track(const json& request_id, const json& params) {
    join_or_reject_active("An operation is already running");
    require_exact_keys(params, {"operation_id", "sample_rate"});
    const std::string operation_id = required_operation_id(params);
    const std::uint32_t sample_rate =
        positive_sample_rate(params.at("sample_rate"));
    std::promise<void> start_gate;
    std::future<void> start_signal = start_gate.get_future();
    backend_.track_start(TrackRequest{sample_rate});
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_operation_id_ = operation_id;
      operation_kind_ = OperationKind::Track;
      track_phase_ = TrackPhase::Uploading;
      latest_track_revision_.store(0, std::memory_order_release);
    }
    canceled_.store(false, std::memory_order_release);
    try {
      operation_thread_ = std::thread(
          [this, operation_id,
           start_signal = std::move(start_signal)]() mutable {
            track_loop(operation_id, std::move(start_signal));
          });
    } catch (...) {
      backend_.abort_operation();
      finish_active();
      throw;
    }
    try {
      emitter_.response(request_id, json::object());
      start_gate.set_value();
    } catch (...) {
      start_gate.set_exception(std::current_exception());
      operation_thread_.join();
      throw;
    }
  }

  void enqueue_track_chunk(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id", "audio_f32le_base64"});
    const std::string operation_id = required_operation_id(params);
    std::vector<float> audio = decode_f32le_base64(
        params.at("audio_f32le_base64"), kMaximumTrackChunkSamples);
    TrackCommand command{TrackCommand::Kind::Chunk, request_id,
                         std::move(audio), 0, {}, std::nullopt};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_track_locked(operation_id);
      if (track_phase_ != TrackPhase::Uploading) {
        throw WorkerError("invalid_state",
                          "Track audio upload is already complete");
      }
      constexpr std::size_t maximum_queued =
          kMaximumTrackChunkSamples * 2U;
      if (command.audio.size() > maximum_queued - track_queued_samples_) {
        throw WorkerError("backpressure", "Track PCM queue is full",
                          {{"maximum_queued_samples", maximum_queued}});
      }
      track_queued_samples_ += command.audio.size();
      track_queue_.push_back(std::move(command));
      pending_operation_request_ids_.insert(request_id.get<std::string>());
    }
    operation_condition_.notify_one();
  }

  void enqueue_track_prepare(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id"});
    const std::string operation_id = required_operation_id(params);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_track_locked(operation_id);
      if (track_phase_ != TrackPhase::Uploading) {
        throw WorkerError("invalid_state", "track_prepare is already queued");
      }
      track_phase_ = TrackPhase::Preparing;
      track_queue_.push_back(
          TrackCommand{TrackCommand::Kind::Prepare, request_id, {}, 0, {},
                       std::nullopt});
      pending_operation_request_ids_.insert(request_id.get<std::string>());
    }
    operation_condition_.notify_one();
  }

  void enqueue_track_render(const json& request_id, const json& params) {
    require_exact_keys(
        params,
        {"operation_id", "revision", "settings_timeline", "preview_sample"});
    const std::string operation_id = required_operation_id(params);
    std::optional<std::int64_t> preview_sample;
    if (!params.at("preview_sample").is_null()) {
      preview_sample = required_nonnegative_int64(
          params.at("preview_sample"), "preview_sample");
    }
    const std::uint64_t revision =
        required_positive_revision(params.at("revision"));
    TrackCommand command{
        TrackCommand::Kind::Render,
        request_id,
        {},
        revision,
        parse_settings_timeline(params.at("settings_timeline")),
        preview_sample};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_track_locked(operation_id);
      if (track_phase_ != TrackPhase::Prepared) {
        throw WorkerError("invalid_state", "track_prepare has not completed");
      }
      const std::uint64_t latest =
          latest_track_revision_.load(std::memory_order_acquire);
      if (revision <= latest) {
        emitter_.response(request_id,
                          track_render_response(revision, 0, true));
        return;
      }

      latest_track_revision_.store(revision, std::memory_order_release);
      for (auto it = track_queue_.begin(); it != track_queue_.end();) {
        if (it->kind != TrackCommand::Kind::Render) {
          ++it;
          continue;
        }
        emitter_.response(it->request_id,
                          track_render_response(it->revision, 0, true));
        retire_operation_request_locked(it->request_id);
        it = track_queue_.erase(it);
      }
      track_queue_.push_back(std::move(command));
      pending_operation_request_ids_.insert(request_id.get<std::string>());
    }
    operation_condition_.notify_one();
  }

  void cancel_operation(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id"});
    const std::string requested = required_operation_id(params);
    std::promise<void> response_gate;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_operation_id_.has_value() ||
          requested != *current_operation_id_ ||
          canceled_.load(std::memory_order_acquire) ||
          terminal_committed_) {
        throw WorkerError("operation_not_found",
                          "The requested operation is not active",
                          {{"operation_id", requested}});
      }
      cancel_response_signal_ = response_gate.get_future().share();
      canceled_.store(true, std::memory_order_release);
    }
    operation_condition_.notify_all();
    respond_and_release(request_id, response_gate);
  }

  void respond_and_release(const json& request_id,
                           std::promise<void>& response_gate) {
    try {
      emitter_.response(request_id, json::object());
      response_gate.set_value();
    } catch (...) {
      response_gate.set_exception(std::current_exception());
      throw;
    }
  }

  void retire_operation_request_locked(const json& request_id) {
    const auto pending = pending_operation_request_ids_.find(
        request_id.get_ref<const std::string&>());
    if (pending != pending_operation_request_ids_.end()) {
      pending_operation_request_ids_.erase(pending);
    }
  }

  void respond_to_active_track(const std::string& operation_id,
                               const json& request_id,
                               json result) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Track was stopped");
    }
    if (!current_operation_id_ || *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Track || terminal_committed_) {
      throw WorkerError("operation_not_found", "The track is no longer active",
                        {{"operation_id", operation_id}});
    }
    // Keep the operation-state lock through the write. A cancel accepted before
    // this point suppresses the stale response; one accepted after it is
    // unambiguously ordered after the response.
    emitter_.response(request_id, std::move(result));
    retire_operation_request_locked(request_id);
  }

  void respond_to_active_stream_settings(const std::string& operation_id,
                                          const std::vector<json>& request_ids) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    if (!current_operation_id_ || *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Stream || terminal_committed_) {
      throw WorkerError("operation_not_found", "The stream is no longer active",
                        {{"operation_id", operation_id}});
    }
    for (const json& request_id : request_ids) {
      emitter_.response(request_id, json::object());
      retire_operation_request_locked(request_id);
    }
  }

  void respond_to_track_render(const std::string& operation_id,
                               const json& request_id,
                               std::uint64_t revision,
                               std::size_t frame_count) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Track was stopped");
    }
    if (!current_operation_id_ || *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Track || terminal_committed_) {
      throw WorkerError("operation_not_found", "The track is no longer active",
                        {{"operation_id", operation_id}});
    }
    const bool superseded =
        frame_count == 0 ||
        latest_track_revision_.load(std::memory_order_acquire) != revision;
    emitter_.response(
        request_id,
        track_render_response(revision,
                              superseded ? 0 : frame_count,
                              superseded));
    retire_operation_request_locked(request_id);
  }

  bool emit_active_track_revision_event(const std::string& operation_id,
                                        std::uint64_t revision,
                                        const char* event,
                                        json data) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire) || terminal_committed_) {
      throw WorkerError("canceled", "Track was stopped");
    }
    if (!current_operation_id_ || *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Track) {
      throw WorkerError("operation_not_found", "The track is no longer active",
                        {{"operation_id", operation_id}});
    }
    if (latest_track_revision_.load(std::memory_order_acquire) != revision) {
      return false;
    }
    emitter_.event(event, std::move(data), operation_id);
    return true;
  }

  void emit_active_stream_event(const std::string& operation_id,
                                const char* event, json data) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire) || terminal_committed_) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    emitter_.event(event, std::move(data), operation_id);
  }

  void stream_loop(const std::string& operation_id,
                   std::future<void> start_signal) {
    try {
      start_signal.get();
    } catch (...) {
      backend_.abort_operation();
      finish_active();
      return;
    }
    const auto emit_frame = [this, &operation_id](const StreamFrame& frame) {
      emit_active_stream_event(
          operation_id, "stream_frame",
          {{"timestamp_sample", frame.timestamp_sample},
           {"weights", frame.weights},
           {"effective_emotions", frame.effective_emotions}});
    };
    try {
      while (true) {
        std::optional<StreamCommand> command;
        std::vector<json> settings_request_ids;
        json latest_settings;
        {
          std::unique_lock<std::mutex> lock(state_mutex_);
          operation_condition_.wait(lock, [this] {
            return canceled_.load(std::memory_order_acquire) ||
                   !stream_queue_.empty();
          });
          if (canceled_.load(std::memory_order_acquire)) break;
          for (auto it = stream_queue_.begin(); it != stream_queue_.end();) {
            if (it->kind != StreamCommand::Kind::Settings) {
              ++it;
              continue;
            }
            settings_request_ids.push_back(std::move(it->request_id));
            latest_settings = std::move(it->settings);
            it = stream_queue_.erase(it);
          }
          if (!stream_queue_.empty()) {
            command.emplace(std::move(stream_queue_.front()));
            stream_queue_.pop_front();
            stream_queued_samples_ -= command->audio.size();
          }
        }
        if (!settings_request_ids.empty()) {
          backend_.stream_settings(latest_settings, canceled_);
          respond_to_active_stream_settings(operation_id,
                                            settings_request_ids);
        }
        if (!command.has_value()) continue;
        command->response_gate.get();
        if (canceled_.load(std::memory_order_acquire)) break;
        if (command->kind == StreamCommand::Kind::Chunk) {
          emit_active_stream_event(operation_id, "stream_credit", json::object());
          backend_.stream_chunk(command->audio, canceled_, emit_frame);
          continue;
        }
        backend_.stream_end(canceled_, emit_frame);
        emit_stream_ended(operation_id);
        finish_active();
        return;
      }
      backend_.abort_operation();
      emit_stream_ended(operation_id);
    } catch (const WorkerError& error) {
      backend_.abort_operation();
      if (error.code() == "canceled") {
        emit_stream_ended(operation_id);
      } else {
        emit_stream_error_or_ended(operation_id, error.code(), error.what());
      }
    } catch (const std::exception& error) {
      backend_.abort_operation();
      emit_stream_error_or_ended(operation_id, "internal_error", error.what());
    }
    finish_active();
  }

  void track_loop(const std::string& operation_id,
                  std::future<void> start_signal) {
    try {
      start_signal.get();
    } catch (...) {
      backend_.abort_operation();
      finish_active();
      return;
    }
    try {
      while (true) {
        TrackCommand command;
        {
          std::unique_lock<std::mutex> lock(state_mutex_);
          operation_condition_.wait(lock, [this] {
            return canceled_.load(std::memory_order_acquire) ||
                   !track_queue_.empty();
          });
          if (canceled_.load(std::memory_order_acquire)) break;
          command = std::move(track_queue_.front());
          track_queue_.pop_front();
          track_queued_samples_ -= command.audio.size();
        }
        if (canceled_.load(std::memory_order_acquire)) break;
        if (command.kind == TrackCommand::Kind::Chunk) {
          backend_.track_chunk(command.audio, canceled_);
          respond_to_active_track(operation_id, command.request_id,
                                  json::object());
          continue;
        }
        if (command.kind == TrackCommand::Kind::Prepare) {
          backend_.track_prepare(canceled_);
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (canceled_.load(std::memory_order_acquire)) {
              throw WorkerError("canceled", "Track was stopped");
            }
            if (!current_operation_id_ ||
                *current_operation_id_ != operation_id ||
                operation_kind_ != OperationKind::Track ||
                terminal_committed_) {
              throw WorkerError("operation_not_found",
                                "The track is no longer active",
                                {{"operation_id", operation_id}});
            }
            track_phase_ = TrackPhase::Prepared;
            emitter_.response(command.request_id, json::object());
            retire_operation_request_locked(command.request_id);
          }
          continue;
        }
        if (command.kind == TrackCommand::Kind::Render) {
          if (latest_track_revision_.load(std::memory_order_acquire) !=
              command.revision) {
            respond_to_active_track(
                operation_id, command.request_id,
                track_render_response(command.revision, 0, true));
            continue;
          }
          const auto emit_preview =
              [this, &operation_id,
               revision = command.revision](const StreamFrame& frame) {
                (void)emit_active_track_revision_event(
                    operation_id, revision, "track_preview",
                    {{"revision", revision},
                     {"timestamp_sample", frame.timestamp_sample},
                     {"weights", frame.weights},
                     {"effective_emotions", frame.effective_emotions}});
              };
          const auto emit_cache =
              [this, &operation_id,
               revision = command.revision](
                  const std::vector<StreamFrame>& frames) {
                for (std::size_t offset = 0; offset < frames.size();
                     offset += kMaximumTrackFramesPerBatch) {
                  const std::size_t end = std::min(
                      frames.size(), offset + kMaximumTrackFramesPerBatch);
                  json timestamp_samples = json::array();
                  json weights = json::array();
                  json effective_emotions = json::array();
                  for (std::size_t index = offset; index < end; ++index) {
                    timestamp_samples.push_back(
                        frames[index].timestamp_sample);
                    weights.push_back(frames[index].weights);
                    effective_emotions.push_back(
                        frames[index].effective_emotions);
                  }
                  if (!emit_active_track_revision_event(
                          operation_id, revision, "track_frame_batch",
                          {{"revision", revision},
                           {"offset", offset},
                           {"total_frames", frames.size()},
                           {"timestamp_samples", std::move(timestamp_samples)},
                           {"weights", std::move(weights)},
                           {"effective_emotions",
                            std::move(effective_emotions)}})) {
                    return;
                  }
                }
              };
          const std::size_t frame_count = backend_.track_render(
              TrackRenderRequest{command.revision,
                                 std::move(command.settings_timeline),
                                 command.preview_sample},
              canceled_, latest_track_revision_, emit_preview, emit_cache);
          respond_to_track_render(operation_id, command.request_id,
                                  command.revision, frame_count);
          continue;
        }
      }
      backend_.abort_operation();
      emit_track_ended(operation_id);
    } catch (const WorkerError& error) {
      backend_.abort_operation();
      if (error.code() == "canceled" ||
          canceled_.load(std::memory_order_acquire)) {
        emit_track_ended(operation_id);
      } else {
        emit_operation_error(operation_id, error.code(), error.what());
      }
    } catch (const std::exception& error) {
      backend_.abort_operation();
      if (canceled_.load(std::memory_order_acquire)) {
        emit_track_ended(operation_id);
      } else {
        emit_operation_error(operation_id, "internal_error", error.what());
      }
    }
    finish_active();
  }

  void emit_track_ended(const std::string& operation_id) {
    std::shared_future<void> signal;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      terminal_committed_ = true;
      if (canceled_.load(std::memory_order_acquire)) {
        signal = cancel_response_signal_;
      }
    }
    if (signal.valid()) {
      try {
        signal.get();
      } catch (...) {
      }
    }
    reject_pending_operation_requests(
        "operation_not_found",
        "The operation ended before the request was processed");
    emitter_.event("track_ended", {{"reason", "canceled"}}, operation_id);
  }

  void emit_operation_error(const std::string& operation_id,
                            const std::string& code,
                            const std::string& message) {
    bool canceled = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      canceled = canceled_.load(std::memory_order_acquire);
      if (!canceled) terminal_committed_ = true;
    }
    if (canceled) {
      emit_track_ended(operation_id);
      return;
    }
    reject_pending_operation_requests(code, message);
    emitter_.event("error", {{"code", code}, {"message", message}},
                   operation_id);
  }

  void emit_stream_ended(const std::string& operation_id) {
    std::shared_future<void> signal;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      terminal_committed_ = true;
      if (canceled_.load(std::memory_order_acquire)) {
        signal = cancel_response_signal_;
      }
    }
    if (signal.valid()) {
      try {
        signal.get();
      } catch (...) {
      }
    }
    reject_pending_operation_requests(
        "operation_not_found",
        "The operation ended before the request was processed");
    emitter_.event("stream_ended", json::object(), operation_id);
  }

  void emit_stream_error_or_ended(const std::string& operation_id,
                                  const std::string& code,
                                  const std::string& message) {
    bool canceled = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      canceled = canceled_.load(std::memory_order_acquire);
      if (!canceled) terminal_committed_ = true;
    }
    if (canceled) {
      emit_stream_ended(operation_id);
      return;
    }
    reject_pending_operation_requests(code, message);
    emitter_.event("error", {{"code", code}, {"message", message}},
                   operation_id);
  }

  void require_stream_locked(const std::string& operation_id) const {
    if (!current_operation_id_.has_value() ||
        *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Stream ||
        canceled_.load(std::memory_order_acquire) || terminal_committed_) {
      throw WorkerError("operation_not_found", "The requested stream is not active",
                        {{"operation_id", operation_id}});
    }
  }

  void require_track_locked(const std::string& operation_id) const {
    if (!current_operation_id_.has_value() ||
        *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Track ||
        canceled_.load(std::memory_order_acquire) || terminal_committed_) {
      throw WorkerError("operation_not_found", "The requested track is not active",
                        {{"operation_id", operation_id}});
    }
  }

  void join_or_reject_active(const char* message) {
    if (operation_thread_.joinable()) {
      bool active;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        active = current_operation_id_.has_value();
      }
      if (active) {
        throw WorkerError("busy", message);
      }
      operation_thread_.join();
    }
  }

  void require_negotiated() const {
    if (!negotiated_) {
      throw WorkerError("invalid_state",
                        "hello must succeed before this method");
    }
  }

  void reject_pending_operation_requests(const std::string& code,
                                          const std::string& message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const std::string& request_id : pending_operation_request_ids_) {
      emitter_.error(json(request_id), code, message);
    }
    pending_operation_request_ids_.clear();
  }

  void finish_active() {
    reject_pending_operation_requests(
        "operation_not_found",
        "The operation ended before the request was processed");
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_operation_id_.reset();
    operation_kind_ = OperationKind::None;
    stream_sample_rate_ = 0;
    stream_end_queued_ = false;
    terminal_committed_ = false;
    stream_queue_.clear();
    stream_queued_samples_ = 0;
    cancel_response_signal_ = {};
    track_phase_ = TrackPhase::None;
    track_queue_.clear();
    track_queued_samples_ = 0;
    latest_track_revision_.store(0, std::memory_order_release);
  }

  void stop_operation() {
    if (operation_thread_.joinable()) {
      canceled_.store(true, std::memory_order_release);
      operation_condition_.notify_all();
      operation_thread_.join();
    }
  }

  Emitter emitter_;
  Backend backend_;
  std::atomic_bool canceled_{false};
  mutable std::mutex state_mutex_;
  std::condition_variable operation_condition_;
  std::optional<std::string> current_operation_id_;
  std::multiset<std::string> pending_operation_request_ids_;
  OperationKind operation_kind_{OperationKind::None};
  std::uint32_t stream_sample_rate_{0};
  std::deque<StreamCommand> stream_queue_;
  std::size_t stream_queued_samples_{0};
  bool stream_end_queued_{false};
  bool terminal_committed_{false};
  std::shared_future<void> cancel_response_signal_;
  TrackPhase track_phase_{TrackPhase::None};
  std::deque<TrackCommand> track_queue_;
  std::size_t track_queued_samples_{0};
  std::atomic<std::uint64_t> latest_track_revision_{0};
  std::thread operation_thread_;
  bool shutting_down_{false};
  bool negotiated_{false};
};

}  // namespace

int run_protocol_server() {
  try {
    Server server;
    return server.run();
  } catch (const std::exception& error) {
    std::cerr << "Fatal worker error: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace a2f_worker
