#include "protocol.h"

#include "backend.h"

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

constexpr const char* kProtocol = "audio2face/10";
constexpr std::size_t kMaximumRequestBytes = 1024U * 1024U;
constexpr std::size_t kStreamQueueSeconds = 4;
constexpr std::size_t kMaximumBakeChunkSamples = 64U * 1024U;

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

struct BakeCommand {
  enum class Kind { Chunk, Prepare, Frame, End };
  Kind kind;
  json request_id;
  std::vector<float> audio;
  std::int64_t target_sample{0};
  json settings;
};

enum class OperationKind { None, Stream, Bake };

enum class BakePhase { None, Uploading, Preparing, Prepared, Ending };

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
          id, {{"worker_profile", "nvidia-a2f3-a2e3-gpu-arkit52/10"},
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
    if (method == "bake_start") {
      require_negotiated();
      start_bake(id, params);
      return;
    }
    if (method == "bake_chunk") {
      require_negotiated();
      enqueue_bake_chunk(id, params);
      return;
    }
    if (method == "bake_prepare") {
      require_negotiated();
      enqueue_bake_prepare(id, params);
      return;
    }
    if (method == "bake_frame") {
      require_negotiated();
      enqueue_bake_frame(id, params);
      return;
    }
    if (method == "bake_end") {
      require_negotiated();
      enqueue_bake_end(id, params);
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
      backend_.stream_abort();
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
    StreamCommand command{StreamCommand::Kind::Chunk, {}, std::move(audio),
                          {}, response_gate.get_future()};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(requested_operation_id);
      const std::size_t maximum_queued =
          static_cast<std::size_t>(stream_sample_rate_) * kStreamQueueSeconds;
      if (command.audio.size() > maximum_queued - stream_queued_samples_) {
        throw WorkerError("stream_backpressure",
                          "Streaming PCM queue is full",
                          {{"maximum_queued_samples", maximum_queued}});
      }
      const std::size_t command_samples = command.audio.size();
      stream_queue_.push_back(std::move(command));
      stream_queued_samples_ += command_samples;
    }
    stream_condition_.notify_one();
    respond_and_release(request_id, response_gate);
  }

  void enqueue_stream_settings(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id", "settings"});
    const std::string requested_operation_id = required_operation_id(params);
    if (!params.at("settings").is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    StreamCommand command{StreamCommand::Kind::Settings, request_id, {},
                          params.at("settings"), {}};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(requested_operation_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
      if (stream_settings_pending_) {
        throw WorkerError("stream_backpressure",
                          "Wait for the pending stream settings response");
      }
      stream_settings_pending_ = true;
      stream_queue_.push_back(std::move(command));
    }
    stream_condition_.notify_one();
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
    stream_condition_.notify_one();
    respond_and_release(request_id, response_gate);
  }

  void start_bake(const json& request_id, const json& params) {
    join_or_reject_active("An operation is already running");
    require_exact_keys(params, {"operation_id", "sample_rate"});
    const std::string operation_id = required_operation_id(params);
    const std::uint32_t sample_rate =
        positive_sample_rate(params.at("sample_rate"));
    std::promise<void> start_gate;
    std::future<void> start_signal = start_gate.get_future();
    json response = backend_.bake_start(BakeRequest{sample_rate});
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_operation_id_ = operation_id;
      operation_kind_ = OperationKind::Bake;
      bake_phase_ = BakePhase::Uploading;
    }
    canceled_.store(false, std::memory_order_release);
    try {
      operation_thread_ = std::thread(
          [this, operation_id,
           start_signal = std::move(start_signal)]() mutable {
            bake_loop(operation_id, std::move(start_signal));
          });
    } catch (...) {
      backend_.stream_abort();
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

  void enqueue_bake_chunk(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id", "audio_f32le_base64"});
    const std::string operation_id = required_operation_id(params);
    std::vector<float> audio = decode_f32le_base64(
        params.at("audio_f32le_base64"), kMaximumBakeChunkSamples);
    BakeCommand command{BakeCommand::Kind::Chunk, request_id,
                        std::move(audio), 0, {}};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_bake_locked(operation_id);
      if (bake_phase_ != BakePhase::Uploading) {
        throw WorkerError("invalid_state",
                          "Bake audio upload is already complete");
      }
      constexpr std::size_t maximum_queued =
          kMaximumBakeChunkSamples * 2U;
      if (command.audio.size() > maximum_queued - bake_queued_samples_) {
        throw WorkerError("stream_backpressure", "Bake PCM queue is full",
                          {{"maximum_queued_samples", maximum_queued}});
      }
      bake_queued_samples_ += command.audio.size();
      bake_queue_.push_back(std::move(command));
    }
    stream_condition_.notify_one();
  }

  void enqueue_bake_prepare(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id"});
    const std::string operation_id = required_operation_id(params);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_bake_locked(operation_id);
      if (bake_phase_ != BakePhase::Uploading) {
        throw WorkerError("invalid_state", "bake_prepare is already queued");
      }
      bake_phase_ = BakePhase::Preparing;
      bake_queue_.push_back(
          BakeCommand{BakeCommand::Kind::Prepare, request_id, {}, 0, {}});
    }
    stream_condition_.notify_one();
  }

  void enqueue_bake_frame(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id", "target_sample", "settings"});
    const std::string operation_id = required_operation_id(params);
    if (!params.at("settings").is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    BakeCommand command{
        BakeCommand::Kind::Frame,
        request_id,
        {},
        required_nonnegative_int64(params.at("target_sample"),
                                   "target_sample"),
        params.at("settings")};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_bake_locked(operation_id);
      if (bake_phase_ != BakePhase::Prepared) {
        throw WorkerError("invalid_state", "bake_prepare has not completed");
      }
      if (bake_frame_pending_) {
        throw WorkerError("stream_backpressure",
                          "Wait for the pending bake frame response");
      }
      bake_frame_pending_ = true;
      bake_queue_.push_back(std::move(command));
    }
    stream_condition_.notify_one();
  }

  void enqueue_bake_end(const json& request_id, const json& params) {
    require_exact_keys(params, {"operation_id"});
    const std::string operation_id = required_operation_id(params);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_bake_locked(operation_id);
      if (bake_phase_ != BakePhase::Prepared) {
        throw WorkerError("invalid_state", "Bake cannot end in its current phase");
      }
      if (bake_frame_pending_) {
        throw WorkerError("stream_backpressure",
                          "Wait for the pending bake frame response");
      }
      bake_phase_ = BakePhase::Ending;
      bake_queue_.push_back(
          BakeCommand{BakeCommand::Kind::End, request_id, {}, 0, {}});
    }
    stream_condition_.notify_one();
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
    stream_condition_.notify_all();
    backend_.interrupt_operation();
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

  void respond_to_active_bake(const std::string& operation_id,
                              const json& request_id,
                              json result,
                              bool completes_frame = false) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Bake was stopped");
    }
    if (!current_operation_id_ || *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Bake || terminal_committed_) {
      throw WorkerError("operation_not_found", "The bake is no longer active",
                        {{"operation_id", operation_id}});
    }
    // Keep the operation-state lock through the write. A cancel accepted before
    // this point suppresses the stale response; one accepted after it is
    // unambiguously ordered after the response.
    emitter_.response(request_id, std::move(result));
    if (completes_frame) bake_frame_pending_ = false;
  }

  void respond_to_active_stream_settings(const std::string& operation_id,
                                         const json& request_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (canceled_.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    if (!current_operation_id_ || *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Stream || terminal_committed_) {
      throw WorkerError("operation_not_found", "The stream is no longer active",
                        {{"operation_id", operation_id}});
    }
    emitter_.response(request_id, json::object());
    stream_settings_pending_ = false;
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
      backend_.stream_abort();
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
        StreamCommand command;
        {
          std::unique_lock<std::mutex> lock(state_mutex_);
          stream_condition_.wait(lock, [this] {
            return canceled_.load(std::memory_order_acquire) ||
                   !stream_queue_.empty();
          });
          if (canceled_.load(std::memory_order_acquire)) break;
          command = std::move(stream_queue_.front());
          stream_queue_.pop_front();
          stream_queued_samples_ -= command.audio.size();
        }
        if (command.kind != StreamCommand::Kind::Settings) {
          command.response_gate.get();
          if (canceled_.load(std::memory_order_acquire)) break;
        }
        if (command.kind == StreamCommand::Kind::Chunk) {
          emit_active_stream_event(operation_id, "stream_credit", json::object());
          backend_.stream_chunk(command.audio, canceled_, emit_frame);
          continue;
        }
        if (command.kind == StreamCommand::Kind::Settings) {
          backend_.stream_settings(command.settings, canceled_);
          respond_to_active_stream_settings(operation_id,
                                            command.request_id);
          continue;
        }
        backend_.stream_end(canceled_, emit_frame);
        emit_stream_ended(operation_id);
        finish_active();
        return;
      }
      backend_.stream_abort();
      emit_stream_ended(operation_id);
    } catch (const WorkerError& error) {
      backend_.stream_abort();
      if (error.code() == "canceled") {
        emit_stream_ended(operation_id);
      } else {
        emit_stream_error_or_ended(operation_id, error.code(), error.what());
      }
    } catch (const std::exception& error) {
      backend_.stream_abort();
      emit_stream_error_or_ended(operation_id, "internal_error", error.what());
    }
    finish_active();
  }

  void bake_loop(const std::string& operation_id,
                 std::future<void> start_signal) {
    try {
      start_signal.get();
    } catch (...) {
      backend_.stream_abort();
      finish_active();
      return;
    }
    try {
      while (true) {
        BakeCommand command;
        {
          std::unique_lock<std::mutex> lock(state_mutex_);
          stream_condition_.wait(lock, [this] {
            return canceled_.load(std::memory_order_acquire) ||
                   !bake_queue_.empty();
          });
          if (canceled_.load(std::memory_order_acquire)) break;
          command = std::move(bake_queue_.front());
          bake_queue_.pop_front();
          bake_queued_samples_ -= command.audio.size();
        }
        if (canceled_.load(std::memory_order_acquire)) break;
        if (command.kind == BakeCommand::Kind::Chunk) {
          respond_to_active_bake(
              operation_id, command.request_id,
              backend_.bake_chunk(command.audio, canceled_));
          continue;
        }
        if (command.kind == BakeCommand::Kind::Prepare) {
          json result = backend_.bake_prepare(canceled_);
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (canceled_.load(std::memory_order_acquire)) {
              throw WorkerError("canceled", "Bake was stopped");
            }
            if (!current_operation_id_ ||
                *current_operation_id_ != operation_id ||
                operation_kind_ != OperationKind::Bake ||
                terminal_committed_) {
              throw WorkerError("operation_not_found",
                                "The bake is no longer active",
                                {{"operation_id", operation_id}});
            }
            bake_phase_ = BakePhase::Prepared;
            emitter_.response(command.request_id, std::move(result));
          }
          continue;
        }
        if (command.kind == BakeCommand::Kind::Frame) {
          const BakeFrame result = backend_.bake_frame(
              BakeFrameRequest{command.target_sample, std::move(command.settings)},
              canceled_);
          respond_to_active_bake(
              operation_id, command.request_id,
              {{"weights", result.weights}},
              true);
          continue;
        }

        if (canceled_.load(std::memory_order_acquire)) {
          throw WorkerError("canceled", "Bake was stopped");
        }
        backend_.bake_end();
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          if (canceled_.load(std::memory_order_acquire)) {
            throw WorkerError("canceled", "Bake was stopped");
          }
          if (!current_operation_id_ ||
              *current_operation_id_ != operation_id ||
              operation_kind_ != OperationKind::Bake ||
              terminal_committed_) {
            throw WorkerError("operation_not_found",
                              "The bake is no longer active",
                              {{"operation_id", operation_id}});
          }
          terminal_committed_ = true;
          emitter_.response(command.request_id, json::object());
          emitter_.event("bake_ended", {{"reason", "completed"}},
                         operation_id);
        }
        finish_active();
        return;
      }
      backend_.stream_abort();
      emit_bake_ended(operation_id);
    } catch (const WorkerError& error) {
      backend_.stream_abort();
      if (error.code() == "canceled" ||
          canceled_.load(std::memory_order_acquire)) {
        emit_bake_ended(operation_id);
      } else {
        emit_operation_error(operation_id, error.code(), error.what());
      }
    } catch (const std::exception& error) {
      backend_.stream_abort();
      if (canceled_.load(std::memory_order_acquire)) {
        emit_bake_ended(operation_id);
      } else {
        emit_operation_error(operation_id, "internal_error", error.what());
      }
    }
    finish_active();
  }

  void emit_bake_ended(const std::string& operation_id) {
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
    emitter_.event("bake_ended", {{"reason", "canceled"}}, operation_id);
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
      emit_bake_ended(operation_id);
      return;
    }
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

  void require_bake_locked(const std::string& operation_id) const {
    if (!current_operation_id_.has_value() ||
        *current_operation_id_ != operation_id ||
        operation_kind_ != OperationKind::Bake ||
        canceled_.load(std::memory_order_acquire) || terminal_committed_) {
      throw WorkerError("operation_not_found", "The requested bake is not active",
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

  void finish_active() noexcept {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_operation_id_.reset();
    operation_kind_ = OperationKind::None;
    stream_sample_rate_ = 0;
    stream_end_queued_ = false;
    terminal_committed_ = false;
    stream_queue_.clear();
    stream_queued_samples_ = 0;
    stream_settings_pending_ = false;
    cancel_response_signal_ = {};
    bake_phase_ = BakePhase::None;
    bake_queue_.clear();
    bake_queued_samples_ = 0;
    bake_frame_pending_ = false;
  }

  void stop_operation() {
    if (operation_thread_.joinable()) {
      canceled_.store(true, std::memory_order_release);
      backend_.interrupt_operation();
      stream_condition_.notify_all();
      operation_thread_.join();
    }
  }

  Emitter emitter_;
  Backend backend_;
  std::atomic_bool canceled_{false};
  mutable std::mutex state_mutex_;
  std::condition_variable stream_condition_;
  std::optional<std::string> current_operation_id_;
  OperationKind operation_kind_{OperationKind::None};
  std::uint32_t stream_sample_rate_{0};
  std::deque<StreamCommand> stream_queue_;
  std::size_t stream_queued_samples_{0};
  bool stream_settings_pending_{false};
  bool stream_end_queued_{false};
  bool terminal_committed_{false};
  std::shared_future<void> cancel_response_signal_;
  BakePhase bake_phase_{BakePhase::None};
  std::deque<BakeCommand> bake_queue_;
  std::size_t bake_queued_samples_{0};
  bool bake_frame_pending_{false};
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
