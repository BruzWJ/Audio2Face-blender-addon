#include "protocol.h"

#include "backend.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
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

constexpr const char* kProtocol = "audio2face/2";
constexpr std::size_t kMaximumRequestBytes = 1024U * 1024U;
constexpr std::size_t kStreamQueueSeconds = 4;

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
             const std::string& job_id) {
    line({{"protocol", kProtocol},
          {"type", "event"},
          {"event", event_name},
          {"job_id", job_id},
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

std::string required_string(const json& object, const char* name) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_string() ||
      it->get_ref<const std::string&>().empty()) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be a non-empty string");
  }
  return it->get<std::string>();
}

std::string operation_id(const json& params, const char* name) {
  const std::string value = required_string(params, name);
  if (value.size() > 128) {
    throw WorkerError("invalid_params", std::string(name) + " is too long");
  }
  return value;
}

std::string required_absolute_path(const json& object, const char* name) {
  std::string value = required_string(object, name);
  if (value.find('\0') != std::string::npos ||
      !std::filesystem::path(value).is_absolute()) {
    throw WorkerError("invalid_params",
                      std::string(name) + " must be an absolute path");
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

enum class ActiveKind { None, Generate, Stream };

struct StreamCommand {
  enum class Kind { Chunk, End };
  Kind kind;
  std::vector<float> audio;
  std::shared_future<void> response_gate;
};

class Server {
 public:
  ~Server() { stop_job(); }

  int run() {
    std::string line;
    while (!shutting_down_ && std::getline(std::cin, line)) {
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
          id, {{"worker_profile", "nvidia-a2f3-a2e3-gpu-arkit52/1"},
               {"worker_version", A2F_WORKER_VERSION}});
      negotiated_ = true;
      return;
    }
    if (method == "load_model") {
      require_negotiated();
      if (busy_.load(std::memory_order_acquire)) {
        throw WorkerError("busy",
                          "Cannot load a model while an operation is running");
      }
      join_completed_job();
      require_exact_keys(params,
                         {"audio2face_model_path", "audio2emotion_model_path",
                          "identity_index"});
      emitter_.response(
          id, backend_.load_model(
                  ModelRequest{required_absolute_path(
                                   params, "audio2face_model_path"),
                               required_absolute_path(
                                   params, "audio2emotion_model_path"),
                               nonnegative_int(params.at("identity_index"),
                                               "identity_index")}));
      return;
    }
    if (method == "generate") {
      require_negotiated();
      start_generate(id, params);
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
    if (method == "stream_end") {
      require_negotiated();
      enqueue_stream_end(id, params);
      return;
    }
    if (method == "cancel") {
      require_negotiated();
      cancel_operation(id, params);
      return;
    }
    if (method == "shutdown") {
      require_exact_keys(params, {});
      stop_job();
      emitter_.response(id, json::object());
      shutting_down_ = true;
      return;
    }
    throw WorkerError("method_not_found", "Unknown method",
                      {{"method", method}});
  }

  void start_generate(const json& request_id, const json& params) {
    join_or_reject_active("A generation job is already running");
    require_exact_keys(params,
                       {"job_id", "audio_path", "result_path", "settings"});
    if (!params.at("settings").is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    GenerateRequest request{
        operation_id(params, "job_id"),
        required_absolute_path(params, "audio_path"),
        required_absolute_path(params, "result_path"), params.at("settings")};
    std::promise<void> start_gate;
    std::future<void> start_signal = start_gate.get_future();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_job_id_ = request.job_id;
      active_kind_ = ActiveKind::Generate;
      terminal_committed_ = false;
      cancel_response_signal_ = {};
    }
    canceled_.store(false, std::memory_order_release);
    busy_.store(true, std::memory_order_release);
    try {
      job_thread_ = std::thread(
          [this, request = std::move(request),
           start_signal = std::move(start_signal)]() mutable {
            try {
              start_signal.get();
            } catch (...) {
              finish_active();
              return;
            }
            try {
              const auto progress =
                  [this, &request](double value, const std::string& stage) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    if (canceled_.load(std::memory_order_acquire) ||
                        terminal_committed_) {
                      throw WorkerError("canceled", "Generation was stopped");
                    }
                    emitter_.event("progress",
                                   {{"progress", value}, {"stage", stage}},
                                   request.job_id);
                  };
              const auto publish_result =
                  [this, &request](const ResultCommit& commit) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    if (canceled_.load(std::memory_order_acquire)) {
                      throw WorkerError(
                          "canceled",
                          "Generation was canceled before result publication");
                    }
                    if (!busy_.load(std::memory_order_acquire) ||
                        active_kind_ != ActiveKind::Generate ||
                        !current_job_id_.has_value() ||
                        *current_job_id_ != request.job_id ||
                        terminal_committed_) {
                      throw WorkerError(
                          "internal_error",
                          "Generation result publication state is invalid");
                    }
                    commit();
                    terminal_committed_ = true;
                  };
              backend_.generate(request, canceled_, progress, publish_result);
              emitter_.event("result", json::object(), request.job_id);
            } catch (const WorkerError& error) {
              const bool cancel_won =
                  commit_terminal_and_wait_for_cancel_response();
              if (error.code() == "canceled" || cancel_won) {
                emitter_.event("canceled", json::object(), request.job_id);
              } else {
                emitter_.event(
                    "error",
                    {{"code", error.code()}, {"message", error.what()}},
                    request.job_id);
              }
            } catch (const std::exception& error) {
              if (commit_terminal_and_wait_for_cancel_response()) {
                emitter_.event("canceled", json::object(), request.job_id);
              } else {
                emitter_.event(
                    "error",
                    {{"code", "internal_error"}, {"message", error.what()}},
                    request.job_id);
              }
            }
            finish_active();
          });
    } catch (...) {
      finish_active();
      throw;
    }
    try {
      emitter_.response(request_id, json::object());
      start_gate.set_value();
    } catch (...) {
      start_gate.set_exception(std::current_exception());
      job_thread_.join();
      throw;
    }
  }

  void start_stream(const json& request_id, const json& params) {
    join_or_reject_active("An operation is already running");
    require_exact_keys(params,
                       {"stream_id", "sample_rate", "settings"});
    if (!params.at("settings").is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    StreamRequest request{operation_id(params, "stream_id"),
                          positive_sample_rate(params.at("sample_rate")),
                          params.at("settings")};
    auto start_gate = std::make_shared<std::promise<void>>();
    std::shared_future<void> start_signal =
        start_gate->get_future().share();
    std::string thread_stream_id = request.stream_id;
    std::optional<std::string> state_stream_id(request.stream_id);
    json response = backend_.stream_start(request);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_job_id_ = std::move(state_stream_id);
      active_kind_ = ActiveKind::Stream;
      stream_sample_rate_ = request.sample_rate;
      stream_end_queued_ = false;
      terminal_committed_ = false;
      stream_queue_.clear();
      stream_queued_samples_ = 0;
      cancel_response_signal_ = {};
    }
    canceled_.store(false, std::memory_order_release);
    busy_.store(true, std::memory_order_release);
    try {
      job_thread_ = std::thread(
          [this, stream_id = std::move(thread_stream_id),
           start_signal = std::move(start_signal)]() mutable {
            stream_loop(stream_id, std::move(start_signal));
          });
    } catch (...) {
      backend_.stream_abort(request.stream_id);
      finish_active();
      throw;
    }
    try {
      emitter_.response(request_id, std::move(response));
      start_gate->set_value();
    } catch (...) {
      start_gate->set_exception(std::current_exception());
      stream_condition_.notify_all();
      job_thread_.join();
      throw;
    }
  }

  void enqueue_stream_chunk(const json& request_id, const json& params) {
    require_exact_keys(params, {"stream_id", "audio_f32le_base64"});
    const std::string stream_id = operation_id(params, "stream_id");
    std::uint32_t sample_rate = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(stream_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
      sample_rate = stream_sample_rate_;
    }
    std::vector<float> audio =
        decode_f32le_base64(params.at("audio_f32le_base64"), sample_rate);
    auto response_gate = std::make_shared<std::promise<void>>();
    StreamCommand command{StreamCommand::Kind::Chunk, std::move(audio),
                          response_gate->get_future().share()};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(stream_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
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
    try {
      emitter_.response(request_id, json::object());
      response_gate->set_value();
    } catch (...) {
      response_gate->set_exception(std::current_exception());
      throw;
    }
  }

  void enqueue_stream_end(const json& request_id, const json& params) {
    require_exact_keys(params, {"stream_id"});
    const std::string stream_id = operation_id(params, "stream_id");
    auto response_gate = std::make_shared<std::promise<void>>();
    StreamCommand command{StreamCommand::Kind::End, {},
                          response_gate->get_future().share()};
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      require_stream_locked(stream_id);
      if (stream_end_queued_) {
        throw WorkerError("invalid_state", "stream_end is already queued");
      }
      stream_queue_.push_back(std::move(command));
      stream_end_queued_ = true;
    }
    stream_condition_.notify_one();
    try {
      emitter_.response(request_id, json::object());
      response_gate->set_value();
    } catch (...) {
      response_gate->set_exception(std::current_exception());
      throw;
    }
  }

  void cancel_operation(const json& request_id, const json& params) {
    require_exact_keys(params, {"job_id"});
    const std::string requested = operation_id(params, "job_id");
    ActiveKind kind = ActiveKind::None;
    auto response_gate = std::make_shared<std::promise<void>>();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_job_id_.has_value() ||
          !busy_.load(std::memory_order_acquire) ||
          requested != *current_job_id_ ||
          canceled_.load(std::memory_order_acquire) ||
          active_kind_ == ActiveKind::None || terminal_committed_) {
        throw WorkerError("job_not_found", "The requested job is not active",
                          {{"job_id", requested}});
      }
      kind = active_kind_;
      cancel_response_signal_ = response_gate->get_future().share();
      canceled_.store(true, std::memory_order_release);
    }
    backend_.cancel();
    if (kind == ActiveKind::Stream) stream_condition_.notify_all();
    try {
      emitter_.response(request_id, json::object());
      response_gate->set_value();
    } catch (...) {
      response_gate->set_exception(std::current_exception());
      throw;
    }
  }

  void stream_loop(const std::string& stream_id,
                   std::shared_future<void> start_signal) {
    try {
      start_signal.get();
    } catch (...) {
      backend_.stream_abort(stream_id);
      finish_active();
      return;
    }
    const auto emit_frame = [this, &stream_id](const StreamFrame& frame) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (canceled_.load(std::memory_order_acquire) || terminal_committed_) {
        throw WorkerError("canceled", "Stream was stopped");
      }
      emitter_.event("stream_frame",
                     {{"timestamp_sample", frame.timestamp_sample},
                      {"weights", frame.weights}},
                     stream_id);
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
        command.response_gate.get();
        if (canceled_.load(std::memory_order_acquire)) break;
        if (command.kind == StreamCommand::Kind::Chunk) {
          backend_.stream_chunk(stream_id, command.audio, canceled_, emit_frame);
          continue;
        }
        backend_.stream_end(stream_id, canceled_, emit_frame);
        emit_stream_ended(stream_id);
        finish_active();
        return;
      }
      wait_for_cancel_response();
      backend_.stream_abort(stream_id);
      emit_stream_ended(stream_id);
    } catch (const WorkerError& error) {
      backend_.stream_abort(stream_id);
      if (error.code() == "canceled") {
        wait_for_cancel_response();
        emit_stream_ended(stream_id);
      } else {
        emit_stream_error_or_ended(stream_id, error.code(), error.what());
      }
    } catch (const std::exception& error) {
      backend_.stream_abort(stream_id);
      emit_stream_error_or_ended(stream_id, "internal_error", error.what());
    }
    finish_active();
  }

  void wait_for_cancel_response() noexcept {
    std::shared_future<void> signal;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      signal = cancel_response_signal_;
    }
    if (signal.valid()) {
      try {
        signal.get();
      } catch (...) {
      }
    }
  }

  bool commit_terminal_and_wait_for_cancel_response() noexcept {
    std::shared_future<void> signal;
    bool canceled = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      terminal_committed_ = true;
      canceled = canceled_.load(std::memory_order_acquire);
      if (canceled) signal = cancel_response_signal_;
    }
    if (signal.valid()) {
      try {
        signal.get();
      } catch (...) {
      }
    }
    return canceled;
  }

  void emit_stream_ended(const std::string& stream_id) {
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
    emitter_.event("stream_ended", json::object(), stream_id);
  }

  void emit_stream_error_or_ended(const std::string& stream_id,
                                  const std::string& code,
                                  const std::string& message) {
    std::shared_future<void> signal;
    bool canceled = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      terminal_committed_ = true;
      canceled = canceled_.load(std::memory_order_acquire);
      if (canceled) signal = cancel_response_signal_;
    }
    if (canceled) {
      if (signal.valid()) {
        try {
          signal.get();
        } catch (...) {
        }
      }
      emitter_.event("stream_ended", json::object(), stream_id);
      return;
    }
    emitter_.event("error", {{"code", code}, {"message", message}},
                   stream_id);
  }

  void require_stream_locked(const std::string& stream_id) const {
    if (!busy_.load(std::memory_order_acquire) ||
        active_kind_ != ActiveKind::Stream ||
        !current_job_id_.has_value() || *current_job_id_ != stream_id ||
        canceled_.load(std::memory_order_acquire) || terminal_committed_) {
      throw WorkerError("job_not_found", "The requested stream is not active",
                        {{"job_id", stream_id}});
    }
  }

  void join_or_reject_active(const char* message) {
    if (job_thread_.joinable()) {
      if (busy_.load(std::memory_order_acquire)) {
        throw WorkerError("busy", message);
      }
      job_thread_.join();
    }
  }

  void join_completed_job() {
    if (job_thread_.joinable() &&
        !busy_.load(std::memory_order_acquire)) {
      job_thread_.join();
    }
  }

  void require_negotiated() const {
    if (!negotiated_) {
      throw WorkerError("invalid_state",
                        "hello must succeed before this method");
    }
  }

  void finish_active() noexcept {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      current_job_id_.reset();
      active_kind_ = ActiveKind::None;
      stream_sample_rate_ = 0;
      stream_end_queued_ = false;
      terminal_committed_ = false;
      stream_queue_.clear();
      stream_queued_samples_ = 0;
      cancel_response_signal_ = {};
      busy_.store(false, std::memory_order_release);
    }
  }

  void stop_job() {
    if (job_thread_.joinable()) {
      canceled_.store(true, std::memory_order_release);
      backend_.cancel();
      stream_condition_.notify_all();
      job_thread_.join();
    }
    finish_active();
  }

  Emitter emitter_;
  Backend backend_;
  std::atomic_bool busy_{false};
  std::atomic_bool canceled_{false};
  mutable std::mutex state_mutex_;
  std::condition_variable stream_condition_;
  std::optional<std::string> current_job_id_;
  ActiveKind active_kind_{ActiveKind::None};
  std::uint32_t stream_sample_rate_{0};
  std::deque<StreamCommand> stream_queue_;
  std::size_t stream_queued_samples_{0};
  bool stream_end_queued_{false};
  bool terminal_committed_{false};
  std::shared_future<void> cancel_response_signal_;
  std::thread job_thread_;
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
