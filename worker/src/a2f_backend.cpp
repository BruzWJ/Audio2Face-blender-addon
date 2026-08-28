#include "backend.h"

#include "path_contract.h"

#include <audio2emotion/audio2emotion.h>
#include <audio2face/audio2face.h>
#include <audio2x/audio_accumulator.h>
#include <audio2x/cuda_stream.h>
#include <audio2x/cuda_utils.h>
#include <audio2x/emotion_accumulator.h>
#include <audio2x/executor.h>
#include <audio2x/tensor_float.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace a2f_worker {
namespace {

constexpr std::size_t kAudio2EmotionInputWindowSamples = 60000;
constexpr std::size_t kAudio2EmotionInferencesToSkip = 30;
constexpr std::size_t kEyesRotationCount = 6;
constexpr std::size_t kArkit52ChannelCount = 52;
constexpr std::uint32_t kMaximumSupportedSampleRate = 384000;
constexpr std::size_t kDefaultIdentityIndex = 0;
constexpr std::size_t kInteractiveEmotionCountPerBuffer = 300;
constexpr std::size_t kMaximumBakeChunkSamples = 64U * 1024U;

struct ArkitEyeLookIndices {
  std::size_t down_left;
  std::size_t in_left;
  std::size_t out_left;
  std::size_t up_left;
  std::size_t down_right;
  std::size_t in_right;
  std::size_t out_right;
  std::size_t up_right;
};

struct PreferredEmotionSettings {
  std::vector<float> values;
  float strength{};
};

struct GeneratedEmotionSettings {
  float emotion_contrast{};
  std::size_t max_emotions{};
  float live_blend_coefficient{};
  float transition_smoothing{};
};

struct EmotionDriver {
  float emotion_strength{};
  std::optional<GeneratedEmotionSettings> generated;
  std::optional<PreferredEmotionSettings> preferred;
};

struct Audio2FaceSettings {
  float input_strength{0.0F};
  nva2f::AnimatorSkinParams skin{};
  nva2f::AnimatorEyesParams eyes{};
};

struct InferenceSettings {
  Audio2FaceSettings audio2face;
  EmotionDriver emotion_driver;
};

template <class T>
struct SdkDestroyer {
  void operator()(T* value) const noexcept {
    if (value != nullptr) value->Destroy();
  }
};

template <class T>
using SdkPtr = std::unique_ptr<T, SdkDestroyer<T>>;

void sdk_check(const std::error_code& error, const char* operation,
               const char* code = "sdk_error") {
  if (error) {
    throw WorkerError(code, std::string(operation) + " failed",
                      {{"sdk_error", error.message()},
                       {"sdk_error_value", error.value()}});
  }
}

template <class T>
SdkPtr<T> require_sdk_ptr(T* value, const char* operation,
                          const char* code = "sdk_error") {
  if (value == nullptr) {
    throw WorkerError(code, std::string(operation) + " returned null");
  }
  return SdkPtr<T>(value);
}

void require_exact_keys(const json& object,
                        std::initializer_list<const char*> expected,
                        const char* path) {
  bool valid = object.size() == expected.size();
  json names = json::array();
  for (const char* name : expected) {
    names.push_back(name);
    valid = valid && object.contains(name);
  }
  if (!valid) {
    throw WorkerError("invalid_params",
                      std::string(path) + " has unexpected or missing keys",
                      {{"expected", std::move(names)}});
  }
}

float required_float_value(const json& value, const std::string& path) {
  if (!value.is_number_float()) {
    throw WorkerError("invalid_params", path + " must be a JSON float");
  }
  const double parsed = value.get<double>();
  if (!std::isfinite(parsed) || parsed < -std::numeric_limits<float>::max() ||
      parsed > std::numeric_limits<float>::max()) {
    throw WorkerError("invalid_params", path + " must be a finite float");
  }
  return static_cast<float>(parsed);
}

float required_float_in_range(const json& object, const char* name,
                              const char* path, float minimum,
                              float maximum) {
  const auto it = object.find(name);
  if (it == object.end()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " is required");
  }
  const std::string value_path = std::string(path) + name;
  const float value = required_float_value(*it, value_path);
  if (value < minimum || value > maximum) {
    throw WorkerError(
        "invalid_params", value_path + " is outside its supported range",
        {{"minimum", minimum}, {"maximum", maximum}, {"received", value}});
  }
  return value;
}

std::size_t required_size_in_range(const json& object, const char* name,
                                   const char* path, std::size_t minimum,
                                   std::size_t maximum) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_number_integer()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " must be a JSON integer");
  }
  if (it->is_number_unsigned()) {
    const std::uint64_t value = it->get<std::uint64_t>();
    if (value >= minimum && value <= maximum) {
      return static_cast<std::size_t>(value);
    }
    throw WorkerError(
        "invalid_params",
        std::string(path) + name + " is outside its supported range",
        {{"minimum", minimum}, {"maximum", maximum}, {"received", value}});
  }
  const std::int64_t value = it->get<std::int64_t>();
  if (value >= 0 && static_cast<std::uint64_t>(value) >= minimum &&
      static_cast<std::uint64_t>(value) <= maximum) {
    return static_cast<std::size_t>(value);
  }
  throw WorkerError(
      "invalid_params",
      std::string(path) + name + " is outside its supported range",
      {{"minimum", minimum}, {"maximum", maximum}, {"received", value}});
}

void require_audio2face_float(float value, const char* name, float minimum,
                              float maximum, const char* code,
                              const char* path) {
  if (!std::isfinite(value) || value < minimum || value > maximum) {
    throw WorkerError(
        code, std::string(path) + name + " is outside its supported range",
        {{"parameter", name},
         {"minimum", minimum},
         {"maximum", maximum},
         {"received", value}});
  }
}

void validate_audio2face_settings(const Audio2FaceSettings& settings,
                                  const char* code, const char* path) {
  const auto require = [code, path](float value, const char* name,
                                    float minimum, float maximum) {
    require_audio2face_float(value, name, minimum, maximum, code, path);
  };
  require(settings.input_strength, "input_strength", 0.0F, 3.0F);
  require(settings.skin.lowerFaceSmoothing, "lower_face_smoothing", 0.0F,
          0.1F);
  require(settings.skin.upperFaceSmoothing, "upper_face_smoothing", 0.0F,
          0.1F);
  require(settings.skin.lowerFaceStrength, "lower_face_strength", 0.0F, 2.0F);
  require(settings.skin.upperFaceStrength, "upper_face_strength", 0.0F, 2.0F);
  require(settings.skin.faceMaskLevel, "face_mask_level", 0.0F, 1.0F);
  require(settings.skin.faceMaskSoftness, "face_mask_softness", 0.001F, 0.5F);
  require(settings.skin.skinStrength, "skin_strength", 0.0F, 2.0F);
  require(settings.skin.blinkStrength, "blink_strength", 0.0F, 2.0F);
  require(settings.skin.eyelidOpenOffset, "eyelid_open_offset", -1.0F, 1.0F);
  require(settings.skin.lipOpenOffset, "lip_open_offset", -0.2F, 0.2F);
  require(settings.skin.blinkOffset, "blink_offset", 0.0F, 1.0F);
  require(settings.eyes.eyeballsStrength, "eyeballs_strength", 0.0F, 2.0F);
  require(settings.eyes.saccadeStrength, "saccade_strength", 0.0F, 2.0F);
  require(settings.eyes.rightEyeballRotationOffsetX,
          "right_eye_rot_x_offset", -10.0F, 10.0F);
  require(settings.eyes.rightEyeballRotationOffsetY,
          "right_eye_rot_y_offset", -10.0F, 10.0F);
  require(settings.eyes.leftEyeballRotationOffsetX, "left_eye_rot_x_offset",
          -10.0F, 10.0F);
  require(settings.eyes.leftEyeballRotationOffsetY, "left_eye_rot_y_offset",
          -10.0F, 10.0F);
  require(settings.eyes.saccadeSeed, "eye_saccade_seed", 0.0F, 4999.0F);
  if (std::floor(settings.eyes.saccadeSeed) != settings.eyes.saccadeSeed) {
    throw WorkerError(
        code, std::string(path) + "eye_saccade_seed must be an integer",
        {{"received", settings.eyes.saccadeSeed}});
  }
}

json audio2face_settings_json(const Audio2FaceSettings& settings) {
  return {{"input_strength", settings.input_strength},
          {"lower_face_smoothing", settings.skin.lowerFaceSmoothing},
          {"upper_face_smoothing", settings.skin.upperFaceSmoothing},
          {"lower_face_strength", settings.skin.lowerFaceStrength},
          {"upper_face_strength", settings.skin.upperFaceStrength},
          {"face_mask_level", settings.skin.faceMaskLevel},
          {"face_mask_softness", settings.skin.faceMaskSoftness},
          {"skin_strength", settings.skin.skinStrength},
          {"blink_strength", settings.skin.blinkStrength},
          {"eyelid_open_offset", settings.skin.eyelidOpenOffset},
          {"lip_open_offset", settings.skin.lipOpenOffset},
          {"eyeballs_strength", settings.eyes.eyeballsStrength},
          {"saccade_strength", settings.eyes.saccadeStrength},
          {"right_eye_rot_x_offset",
           settings.eyes.rightEyeballRotationOffsetX},
          {"right_eye_rot_y_offset",
           settings.eyes.rightEyeballRotationOffsetY},
          {"left_eye_rot_x_offset", settings.eyes.leftEyeballRotationOffsetX},
          {"left_eye_rot_y_offset", settings.eyes.leftEyeballRotationOffsetY},
          {"eye_saccade_seed",
           static_cast<std::size_t>(settings.eyes.saccadeSeed)}};
}

std::vector<std::string> skin_pose_names(
    const nva2f::BlendshapeSolveExecutorCreationParameters::BlendshapeParams* params) {
  if (params == nullptr) {
    throw WorkerError("model_invalid", "Model has no skin blendshape solver");
  }
  if (params->data.poseNames == nullptr && params->data.poseNamesSize != 0) {
    throw WorkerError("model_invalid", "Blendshape pose names pointer is null");
  }
  std::vector<std::string> names;
  names.reserve(params->data.poseNamesSize);
  for (std::size_t index = 0; index < params->data.poseNamesSize; ++index) {
    const char* name = params->data.poseNames[index];
    if (name == nullptr || *name == '\0') {
      throw WorkerError("model_invalid", "Model contains an empty blendshape name",
                        {{"index", index}});
    }
    names.emplace_back(name);
  }
  return names;
}

void validate_arkit52_channels(const std::vector<std::string>& names) {
  json duplicates = json::array();
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (std::find(names.begin(), names.begin() + index, names[index]) !=
        names.begin() + index) {
      duplicates.push_back(names[index]);
    }
  }
  if (names.size() != kArkit52ChannelCount || !duplicates.empty()) {
    throw WorkerError(
        "model_invalid",
        "Skin blendshape solver is not one unique 52-channel ARKit output",
        {{"reported_count", names.size()},
         {"expected_count", kArkit52ChannelCount},
         {"duplicates", std::move(duplicates)}});
  }
}

std::size_t require_channel_index(const std::vector<std::string>& names,
                                  const char* semantic_name) {
  const auto found = std::find(names.begin(), names.end(), semantic_name);
  if (found == names.end()) {
    throw WorkerError(
        "model_invalid", "ARKit eye-look channel is missing from the model",
        {{"channel", semantic_name}});
  }
  return static_cast<std::size_t>(found - names.begin());
}

ArkitEyeLookIndices resolve_arkit_eye_look_indices(
    const std::vector<std::string>& names) {
  return {require_channel_index(names, "eyeLookDownLeft"),
          require_channel_index(names, "eyeLookInLeft"),
          require_channel_index(names, "eyeLookOutLeft"),
          require_channel_index(names, "eyeLookUpLeft"),
          require_channel_index(names, "eyeLookDownRight"),
          require_channel_index(names, "eyeLookInRight"),
          require_channel_index(names, "eyeLookOutRight"),
          require_channel_index(names, "eyeLookUpRight")};
}

void resolve_arkit_eye_look(std::vector<float>& weights,
                            const float* eyes,
                            const ArkitEyeLookIndices& indices) {
  constexpr float kEyeRangeDegrees = 60.0F;
  const float right_x = eyes[0] / kEyeRangeDegrees;
  const float right_y = eyes[1] / kEyeRangeDegrees;
  const float left_x = eyes[3] / kEyeRangeDegrees;
  const float left_y = eyes[4] / kEyeRangeDegrees;
  weights[indices.down_left] = left_x;
  weights[indices.in_left] = -left_y;
  weights[indices.out_left] = left_y;
  weights[indices.up_left] = -left_x;
  weights[indices.down_right] = right_x;
  weights[indices.in_right] = right_y;
  weights[indices.out_right] = -right_y;
  weights[indices.up_right] = -right_x;
}

}  // namespace

class Backend::Impl final {
 public:
  ~Impl() {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    clear_locked();
  }

  json load_model(const ModelRequest& request) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    if (operation_active_.load(std::memory_order_acquire)) {
      throw WorkerError("busy", "Cannot replace the model during inference");
    }
    clear_locked();
    try {
      require_canonical_regular_file(
          request.audio2face_model_path, "model_not_found",
          "audio2face_model_path");
      require_canonical_regular_file(
          request.audio2emotion_model_path, "model_not_found",
          "audio2emotion_model_path");
      sdk_check(nva2x::SetCudaDeviceIfNeeded(0), "Selecting CUDA device",
                "gpu_error");
      geometry_model_info_ = require_sdk_ptr(
          nva2f::ReadDiffusionModelInfo(
              request.audio2face_model_path.c_str()),
          "Reading diffusion model", "model_invalid");
      const auto& network = geometry_model_info_->GetNetworkInfo();
      const auto& audio2face_network = network.GetNetworkInfo();
      const std::size_t audio2face_input_window_samples =
          audio2face_network.bufferLength;
      if (audio2face_input_window_samples == 0 ||
          audio2face_network.bufferSamplerate == 0) {
        throw WorkerError(
            "model_invalid",
            "Audio2Face reported invalid audio window metadata",
            {{"buffer_samples", audio2face_input_window_samples},
                         {"sample_rate", audio2face_network.bufferSamplerate}});
      }
      if (audio2face_network.bufferSamplerate > kMaximumSupportedSampleRate) {
        throw WorkerError("model_invalid", "SDK reported an invalid sample rate",
                          {{"sample_rate",
                            audio2face_network.bufferSamplerate}});
      }
      sample_rate_ =
          static_cast<std::uint32_t>(audio2face_network.bufferSamplerate);
      prebuffer_samples_ = audio2face_input_window_samples;
      if (network.GetIdentityLength() == 0) {
        throw WorkerError("model_invalid",
                          "Audio2Face model has no identity");
      }
      json emotion_channels =
          load_emotion_channels(network, network.GetDefaultEmotion());

      blendshape_model_info_ = require_sdk_ptr(
          nva2f::ReadDiffusionBlendshapeSolveModelInfo(
              request.audio2face_model_path.c_str()),
          "Reading diffusion blendshape solver", "model_invalid");
      const auto blendshape_parameters =
          blendshape_model_info_->GetExecutorCreationParameters(
              nva2f::IGeometryExecutor::ExecutionOption::Skin,
              kDefaultIdentityIndex);
      std::vector<std::string> output_channels =
          skin_pose_names(blendshape_parameters.initializationSkinParams);
      validate_arkit52_channels(output_channels);
      eye_look_indices_ = resolve_arkit_eye_look_indices(output_channels);

      emotion_model_info_ = require_sdk_ptr(
          nva2e::ReadClassifierModelInfo(
              request.audio2emotion_model_path.c_str()),
          "Reading Audio2Emotion classifier model", "model_invalid");
      cuda_stream_ = require_sdk_ptr(nva2x::CreateCudaStream(),
                                    "Creating CUDA stream", "gpu_error");
      ensure_interactive_executors();

      Audio2FaceSettings audio2face_defaults;
      sdk_check(nva2f::GetInteractiveExecutorInputStrength(
                    *interactive_executor_, audio2face_defaults.input_strength),
                "Reading interactive Audio2Face input strength");
      sdk_check(nva2f::GetInteractiveExecutorSkinParameters(
                    *interactive_executor_, audio2face_defaults.skin),
                "Reading interactive Audio2Face skin parameters");
      sdk_check(nva2f::GetInteractiveExecutorEyesParameters(
                    *interactive_executor_, audio2face_defaults.eyes),
                "Reading interactive Audio2Face eyes parameters");
      validate_audio2face_settings(audio2face_defaults, "model_invalid",
                                   "Audio2Face model tuning default ");
      audio2face_defaults_ = audio2face_defaults;

      json model_schema = {{"channels", std::move(output_channels)},
                           {"emotion_channels", std::move(emotion_channels)},
                           {"audio2face_defaults",
                            audio2face_settings_json(audio2face_defaults_)}};
      return {{"sample_rate", sample_rate_},
              {"model_schema", std::move(model_schema)}};
    } catch (...) {
      clear_locked();
      throw;
    }
  }

  json stream_start(const StreamRequest& request) {
    begin_operation(request.sample_rate, request.settings);
    try {
      return {{"sample_rate", sample_rate_},
              {"prebuffer_samples", prebuffer_samples_}};
    } catch (...) {
      finish_operation();
      throw;
    }
  }

  void stream_chunk(const std::vector<float>& audio,
                    std::atomic_bool& canceled,
                    const StreamFrameCallback& frame) {
    require_active_stream();
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    retain_audio(audio);
    evaluate_interactive_stream(false, canceled, frame);
  }

  void stream_settings(const json& settings,
                       std::atomic_bool& canceled) {
    require_active_stream();
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    InferenceSettings parsed = parse_settings(settings);
    configure_interactive_audio2face(parsed.audio2face);
    interactive_emotion_driver_ = std::move(parsed.emotion_driver);
    if (interactive_emotion_driver_.generated) {
      configure_interactive_generated_emotion();
      interactive_emotions_valid_ = false;
      interactive_emotion_settings_ = json();
    } else {
      install_interactive_driver_emotions();
      interactive_emotion_settings_ = settings.at("emotion_driver");
    }
    interactive_stream_settings_ = settings;
  }

  void stream_end(std::atomic_bool& canceled,
                  const StreamFrameCallback& frame) {
    require_active_stream();
    OperationReset reset(*this);
    evaluate_interactive_stream(true, canceled, frame);
  }

  json bake_start(const BakeRequest& request) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    require_model_locked();
    if (request.sample_rate != sample_rate_) {
      throw WorkerError("sample_rate_mismatch",
                        "Bake PCM must use the model sample rate",
                        {{"expected", sample_rate_},
                         {"received", request.sample_rate}});
    }
    if (operation_active_.exchange(true, std::memory_order_acq_rel)) {
      throw WorkerError("busy", "An Audio2Face operation is already active");
    }
    operation_kind_ = OperationKind::Bake;
    bake_audio_.clear();
    bake_prepared_ = false;
    return {{"sample_rate", sample_rate_},
            {"max_chunk_samples", kMaximumBakeChunkSamples}};
  }

  json bake_chunk(const std::vector<float>& audio,
                  std::atomic_bool& canceled) {
    require_active_bake();
    if (bake_prepared_) {
      throw WorkerError("invalid_state", "bake_prepare is already complete");
    }
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Bake was stopped");
    }
    if (audio.size() > static_cast<std::uint64_t>(
                           std::numeric_limits<std::int64_t>::max()) -
                           bake_audio_.size()) {
      throw WorkerError("invalid_state", "Bake sample timeline is too long");
    }
    bake_audio_.insert(bake_audio_.end(), audio.begin(), audio.end());
    return {{"accepted_samples", audio.size()},
            {"total_samples", bake_audio_.size()}};
  }

  json bake_prepare(std::atomic_bool& canceled) {
    require_active_bake();
    if (bake_prepared_) {
      throw WorkerError("invalid_state", "bake_prepare is already complete");
    }
    if (bake_audio_.empty()) {
      throw WorkerError("invalid_state", "Bake audio is empty");
    }
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Bake was stopped");
    }
    refill_interactive_audio(bake_audio_);
    install_constant_interactive_emotions(
        std::vector<float>(emotion_channels_.size(), 0.0F));
    bake_prepared_ = true;
    const std::size_t source_frame_count =
        interactive_executor_->GetTotalNbFrames();
    if (source_frame_count == 0) {
      throw WorkerError("inference_failed",
                        "Interactive Audio2Face reported no source frames");
    }
    return {{"audio_samples", bake_audio_.size()},
            {"source_frame_count", source_frame_count},
            {"source_frame_rate_numerator", frame_rate_numerator_},
            {"source_frame_rate_denominator", frame_rate_denominator_},
            {"first_source_timestamp_sample",
             interactive_executor_->GetFrameTimestamp(0)},
            {"last_source_timestamp_sample",
             interactive_executor_->GetFrameTimestamp(source_frame_count - 1)},
            {"sampling", "linear"}};
  }

  BakeFrame bake_frame(const BakeFrameRequest& request,
                       std::atomic_bool& canceled) {
    require_active_bake();
    if (!bake_prepared_) {
      throw WorkerError("invalid_state", "bake_prepare must complete first");
    }
    if (request.target_sample < 0 ||
        static_cast<std::uint64_t>(request.target_sample) >=
            bake_audio_.size()) {
      throw WorkerError(
          "invalid_params", "target_sample is outside the bake audio",
          {{"target_sample", request.target_sample},
           {"audio_samples", bake_audio_.size()}});
    }
    prepare_interactive_settings(request.settings, canceled);
    return compute_bake_frame(request, canceled);
  }

  void bake_end() {
    require_active_bake();
    finish_operation();
  }

  void interrupt_operation() noexcept {
    nva2x::IInteractiveExecutor* active = nullptr;
    {
      std::lock_guard<std::mutex> lock(interactive_compute_mutex_);
      active = active_interactive_compute_;
    }
    if (active != nullptr) (void)active->Interrupt();
  }

  void stream_abort() noexcept { finish_operation(); }

 private:
  class OperationReset final {
   public:
    explicit OperationReset(Impl& owner) noexcept : owner_(owner) {}
    OperationReset(const OperationReset&) = delete;
    OperationReset& operator=(const OperationReset&) = delete;
    ~OperationReset() { owner_.finish_operation(); }

   private:
    Impl& owner_;
  };

  enum class OperationKind { None, Stream, Bake };

  struct PendingFrame {
    std::int64_t next_timestamp{0};
    SdkPtr<nva2x::IHostTensorFloat> weights;
    SdkPtr<nva2x::IHostTensorFloat> eyes;
  };

  struct Capture {
    std::atomic_bool* canceled{nullptr};
    std::size_t weight_count{0};
    std::map<std::int64_t, PendingFrame> frames;
    const char* failure{nullptr};
  };

  struct InteractiveEmotionCapture {
    std::atomic_bool* canceled{nullptr};
    nva2x::IEmotionAccumulator* accumulator{nullptr};
    std::size_t emotion_count{0};
    std::map<std::int64_t, SdkPtr<nva2x::IHostTensorFloat>> frames;
    std::error_code accumulation_error;
    const char* failure{nullptr};
  };

  static void fail_capture(Capture& capture, const char* message) noexcept {
    capture.failure = message;
  }

  static bool pending_timestamp_matches(
      const PendingFrame& frame, std::int64_t next_timestamp) noexcept {
    return (!frame.weights && !frame.eyes) ||
           frame.next_timestamp == next_timestamp;
  }

  static bool interactive_emotion_callback(
      void* userdata,
      const nva2e::IEmotionInteractiveExecutor::Results& results) {
    auto& owner = *static_cast<Impl*>(userdata);
    if (owner.active_interactive_emotion_capture_ == nullptr) return false;
    auto& capture = *owner.active_interactive_emotion_capture_;
    if (capture.canceled->load(std::memory_order_acquire)) return false;
    if (results.trackIndex != 0) {
      capture.failure =
          "Interactive Audio2Emotion returned an unexpected track";
      return false;
    }
    if (results.emotions.Size() != capture.emotion_count) {
      capture.failure =
          "Interactive Audio2Emotion returned an unexpected value count";
      return false;
    }
    if (capture.frames.count(results.timeStampCurrentFrame) != 0) {
      capture.failure = "Interactive Audio2Emotion returned a duplicate frame";
      return false;
    }
    capture.accumulation_error = capture.accumulator->Accumulate(
        results.timeStampCurrentFrame, results.emotions, results.cudaStream);
    if (capture.accumulation_error) {
      capture.failure = "Accumulating interactive emotion failed";
      return false;
    }
    try {
      auto host = require_sdk_ptr(
          nva2x::CreateHostPinnedTensorFloat(capture.emotion_count),
          "Allocating pinned interactive emotion buffer", "gpu_error");
      const auto copy_error = nva2x::CopyDeviceToHost(
          host->View(0, capture.emotion_count), results.emotions,
          results.cudaStream);
      if (copy_error) {
        capture.accumulation_error = copy_error;
        capture.failure = "Copying interactive emotion failed";
        return false;
      }
      capture.frames.emplace(results.timeStampCurrentFrame, std::move(host));
      return true;
    } catch (...) {
      capture.failure = "Interactive Audio2Emotion callback failed";
      return false;
    }
  }


  static bool geometry_callback(
      void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& owner = *static_cast<Impl*>(userdata);
    if (owner.active_capture_ == nullptr) return false;
    auto& capture = *owner.active_capture_;
    if (capture.canceled->load(std::memory_order_acquire)) return false;
    try {
      if (results.trackIndex != 0) {
        fail_capture(capture, "Geometry callback returned an unexpected track");
        return false;
      }
      if (results.eyesRotation.Size() != kEyesRotationCount) {
        fail_capture(capture,
                     "Geometry callback returned an unexpected eye rotation count");
        return false;
      }
      auto& frame = capture.frames[results.timeStampCurrentFrame];
      if (frame.eyes ||
          !pending_timestamp_matches(frame, results.timeStampNextFrame)) {
        fail_capture(capture,
                     "Geometry callback returned a duplicate or inconsistent frame");
        return false;
      }
      frame.next_timestamp = results.timeStampNextFrame;
      frame.eyes = require_sdk_ptr(
          nva2x::CreateHostPinnedTensorFloat(kEyesRotationCount),
          "Allocating pinned eye result buffer", "gpu_error");
      const auto error = nva2x::CopyDeviceToHost(
          frame.eyes->View(0, kEyesRotationCount),
          results.eyesRotation, results.eyesCudaStream);
      if (error) {
        fail_capture(capture, "Copying eye rotations failed");
        return false;
      }
      return true;
    } catch (...) {
      fail_capture(capture, "Geometry callback failed");
      return false;
    }
  }

  static bool weights_callback(
      void* userdata,
      const nva2f::IBlendshapeExecutor::DeviceResults& results) {
    auto& owner = *static_cast<Impl*>(userdata);
    if (owner.active_capture_ == nullptr) return false;
    auto& capture = *owner.active_capture_;
    if (capture.canceled->load(std::memory_order_acquire)) return false;
    try {
      if (results.trackIndex != 0) {
        fail_capture(capture, "Blendshape callback returned an unexpected track");
        return false;
      }
      if (results.weights.Size() != capture.weight_count) {
        fail_capture(capture,
                     "Blendshape callback returned an unexpected value count");
        return false;
      }
      auto& frame = capture.frames[results.timeStampCurrentFrame];
      if (frame.weights ||
          !pending_timestamp_matches(frame, results.timeStampNextFrame)) {
        fail_capture(
            capture,
            "Blendshape callback returned a duplicate or inconsistent frame");
        return false;
      }
      frame.next_timestamp = results.timeStampNextFrame;
      frame.weights = require_sdk_ptr(
          nva2x::CreateHostPinnedTensorFloat(capture.weight_count),
          "Allocating pinned blendshape result buffer", "gpu_error");
      const auto error = nva2x::CopyDeviceToHost(
          frame.weights->View(0, capture.weight_count),
          results.weights, results.cudaStream);
      if (error) {
        fail_capture(capture, "Copying blendshape weights failed");
        return false;
      }
      return true;
    } catch (...) {
      fail_capture(capture, "Blendshape callback failed");
      return false;
    }
  }

  nva2f::IGeometryInteractiveExecutor& interactive_geometry_executor() {
    nva2f::IGeometryInteractiveExecutor* geometry = nullptr;
    sdk_check(nva2f::GetInteractiveExecutorGeometryExecutor(
                  *interactive_executor_, &geometry),
              "Retrieving interactive geometry executor");
    if (geometry == nullptr) {
      throw WorkerError("sdk_error", "Interactive geometry executor is null");
    }
    return *geometry;
  }

  void ensure_interactive_executors() {
    if (interactive_executor_ != nullptr) return;
    if (geometry_model_info_ == nullptr || blendshape_model_info_ == nullptr ||
        emotion_model_info_ == nullptr || cuda_stream_ == nullptr) {
      throw WorkerError("model_not_loaded", "Load a model before inference");
    }

    interactive_audio_accumulator_ = require_sdk_ptr(
        nva2x::CreateAudioAccumulator(sample_rate_, 0),
        "Creating interactive audio accumulator", "gpu_error");
    interactive_emotion_accumulator_ = require_sdk_ptr(
        nva2x::CreateEmotionAccumulator(
            emotion_channels_.size(), kInteractiveEmotionCountPerBuffer, 0),
        "Creating interactive emotion accumulator", "gpu_error");

    nva2f::GeometryExecutorCreationParameters geometry_parameters;
    geometry_parameters.cudaStream = cuda_stream_->Data();
    geometry_parameters.nbTracks = 1;
    const nva2x::IAudioAccumulator* audio_accumulator =
        interactive_audio_accumulator_.get();
    geometry_parameters.sharedAudioAccumulators = &audio_accumulator;
    const nva2x::IEmotionAccumulator* emotion_accumulator =
        interactive_emotion_accumulator_.get();
    geometry_parameters.sharedEmotionAccumulators = &emotion_accumulator;
    const auto diffusion_parameters =
        geometry_model_info_->GetExecutorCreationParameters(
            nva2f::IGeometryExecutor::ExecutionOption::All,
            kDefaultIdentityIndex, true);
    auto geometry = require_sdk_ptr(
        nva2f::CreateDiffusionGeometryInteractiveExecutor(
            geometry_parameters, diffusion_parameters, 0),
        "Creating diffusion interactive geometry executor", "gpu_error");

    const auto solve_parameters =
        blendshape_model_info_->GetExecutorCreationParameters(
            nva2f::IGeometryExecutor::ExecutionOption::Skin,
            kDefaultIdentityIndex);
    nva2f::DeviceBlendshapeSolveExecutorCreationParameters blendshape_parameters;
    blendshape_parameters.initializationSkinParams =
        solve_parameters.initializationSkinParams;
    blendshape_parameters.initializationTongueParams =
        solve_parameters.initializationTongueParams;
    interactive_executor_ = require_sdk_ptr(
        nva2f::CreateDeviceBlendshapeSolveInteractiveExecutor(
            geometry.release(), blendshape_parameters),
        "Creating device blendshape interactive executor", "gpu_error");
    if (interactive_executor_->GetResultType() !=
            nva2f::IBlendshapeExecutor::ResultsType::DEVICE ||
        interactive_executor_->GetWeightCount() != kArkit52ChannelCount) {
      throw WorkerError(
          "model_invalid", "Interactive blendshape output is incompatible",
          {{"reported_weights", interactive_executor_->GetWeightCount()},
           {"expected_weights", kArkit52ChannelCount}});
    }
    auto& interactive_geometry = interactive_geometry_executor();
    if (interactive_geometry.GetEyesRotationSize() != kEyesRotationCount) {
      throw WorkerError(
          "model_invalid", "Unsupported interactive eyes rotation size",
          {{"reported", interactive_geometry.GetEyesRotationSize()},
           {"expected", kEyesRotationCount}});
    }
    sdk_check(nva2f::SetInteractiveExecutorGeometryResultsCallback(
                  *interactive_executor_, &Impl::geometry_callback, this),
              "Installing interactive geometry callback");
    sdk_check(interactive_executor_->SetResultsCallback(
                  &Impl::weights_callback, this),
              "Installing interactive blendshape callback");

    interactive_executor_->GetFrameRate(frame_rate_numerator_,
                                        frame_rate_denominator_);
    if (frame_rate_numerator_ == 0 || frame_rate_denominator_ == 0) {
      throw WorkerError("model_invalid",
                        "Audio2Face reported an invalid frame rate");
    }
    const std::size_t timestamp_denominator =
        static_cast<std::size_t>(sample_rate_) * frame_rate_denominator_;
    frame_alignment_samples_ =
        timestamp_denominator /
        std::gcd(timestamp_denominator, frame_rate_numerator_);

    nva2e::EmotionExecutorCreationParameters emotion_parameters;
    emotion_parameters.cudaStream = cuda_stream_->Data();
    emotion_parameters.nbTracks = 1;
    emotion_parameters.sharedAudioAccumulators = &audio_accumulator;
    const auto classifier_parameters =
        emotion_model_info_->GetExecutorCreationParameters(
            kAudio2EmotionInputWindowSamples, frame_rate_numerator_,
            frame_rate_denominator_, kAudio2EmotionInferencesToSkip);
    const std::size_t audio2emotion_input_window_samples =
        classifier_parameters.networkInfo.bufferLength;
    if (audio2emotion_input_window_samples == 0 ||
        classifier_parameters.networkInfo.bufferSamplerate != sample_rate_) {
      throw WorkerError(
          "model_invalid", "Audio2Emotion reported invalid audio window metadata",
          {{"buffer_samples", audio2emotion_input_window_samples},
           {"audio2emotion_sample_rate",
            classifier_parameters.networkInfo.bufferSamplerate},
           {"audio2face_sample_rate", sample_rate_}});
    }
    prebuffer_samples_ =
        std::max(prebuffer_samples_, audio2emotion_input_window_samples);
    interactive_emotion_executor_ = require_sdk_ptr(
        nva2e::CreateClassifierEmotionInteractiveExecutor(
            emotion_parameters, classifier_parameters, 1),
        "Creating Audio2Emotion interactive executor", "gpu_error");
    if (interactive_emotion_executor_->GetSamplingRate() != sample_rate_ ||
        interactive_emotion_executor_->GetEmotionsSize() !=
            emotion_channels_.size()) {
      throw WorkerError(
          "model_invalid", "Interactive Audio2Emotion output is incompatible");
    }
    sdk_check(interactive_emotion_executor_->SetResultsCallback(
                  &Impl::interactive_emotion_callback, this),
              "Installing interactive Audio2Emotion callback");
  }

  void require_model_locked() const {
    if (interactive_executor_ == nullptr) {
      throw WorkerError("model_not_loaded", "Load a model before inference");
    }
  }

  std::error_code compute_interactive(
      nva2x::IInteractiveExecutor& executor,
      std::atomic_bool& canceled,
      std::optional<std::size_t> frame_index = std::nullopt) {
    {
      std::lock_guard<std::mutex> lock(interactive_compute_mutex_);
      if (canceled.load(std::memory_order_acquire)) {
        throw WorkerError("canceled", "Operation was stopped");
      }
      if (active_interactive_compute_ != nullptr) {
        throw WorkerError("invalid_state",
                          "Another interactive computation is already active");
      }
      active_interactive_compute_ = &executor;
    }

    try {
      const std::error_code result = frame_index
                                         ? executor.ComputeFrame(*frame_index)
                                         : executor.ComputeAllFrames();
      std::lock_guard<std::mutex> lock(interactive_compute_mutex_);
      active_interactive_compute_ = nullptr;
      return result;
    } catch (...) {
      std::lock_guard<std::mutex> lock(interactive_compute_mutex_);
      active_interactive_compute_ = nullptr;
      throw;
    }
  }

  void refill_interactive_audio(const std::vector<float>& audio) {
    sdk_check(interactive_audio_accumulator_->Reset(),
              "Resetting interactive audio accumulator");
    sdk_check(interactive_audio_accumulator_->Accumulate(
                  nva2x::HostTensorFloatConstView(audio.data(), audio.size()),
                  cuda_stream_->Data()),
              "Accumulating interactive audio");
    sdk_check(interactive_audio_accumulator_->Close(),
              "Closing interactive audio accumulator");
    sdk_check(interactive_executor_->Invalidate(
                  nva2f::IGeometryInteractiveExecutor::kLayerAudioAccumulator),
              "Invalidating interactive Audio2Face audio");
    sdk_check(interactive_emotion_executor_->Invalidate(
                  nva2e::IEmotionInteractiveExecutor::kLayerAudioAccumulator),
              "Invalidating interactive Audio2Emotion audio");
    interactive_emotions_valid_ = false;
    interactive_emotion_settings_ = json();
  }

  void install_constant_interactive_emotions(
      const std::vector<float>& emotion) {
    sdk_check(interactive_emotion_accumulator_->Reset(),
              "Resetting interactive emotion accumulator");
    sdk_check(interactive_emotion_accumulator_->Accumulate(
                  0,
                  nva2x::HostTensorFloatConstView(emotion.data(),
                                                 emotion.size()),
                  cuda_stream_->Data()),
              "Accumulating constant interactive emotions");
    sdk_check(interactive_emotion_accumulator_->Close(),
              "Closing constant interactive emotions");
    sdk_check(interactive_executor_->Invalidate(
                  nva2f::IGeometryInteractiveExecutor::kLayerEmotionAccumulator),
              "Invalidating interactive Audio2Face emotions");
    interactive_effective_emotions_.clear();
    interactive_effective_emotions_.emplace(0, emotion);
    interactive_emotions_valid_ = true;
  }

  void configure_interactive_audio2face(
      const Audio2FaceSettings& settings) {
    sdk_check(nva2f::SetInteractiveExecutorInputStrength(
                  *interactive_executor_, settings.input_strength),
              "Configuring interactive Audio2Face input strength");
    sdk_check(nva2f::SetInteractiveExecutorSkinParameters(
                  *interactive_executor_, settings.skin),
              "Configuring interactive Audio2Face skin parameters");
    sdk_check(nva2f::SetInteractiveExecutorEyesParameters(
                  *interactive_executor_, settings.eyes),
              "Configuring interactive Audio2Face eyes parameters");
  }

  void install_interactive_driver_emotions() {
    std::vector<float> emotion(emotion_channels_.size(), 0.0F);
    if (interactive_emotion_driver_.preferred) {
      const PreferredEmotionSettings& preferred =
          *interactive_emotion_driver_.preferred;
      const float scale = interactive_emotion_driver_.emotion_strength *
                          preferred.strength;
      std::transform(preferred.values.begin(), preferred.values.end(),
                     emotion.begin(),
                     [scale](float value) { return scale * value; });
    }
    install_constant_interactive_emotions(emotion);
  }

  void configure_interactive_generated_emotion() {
    const GeneratedEmotionSettings& generated =
        interactive_emotion_driver_.generated.value();
    nva2e::PostProcessParams parameters;
    sdk_check(nva2e::GetInteractiveExecutorPostProcessParameters(
                  *interactive_emotion_executor_, parameters),
              "Reading interactive Audio2Emotion post-process parameters");
    parameters.emotionStrength = interactive_emotion_driver_.emotion_strength;
    parameters.emotionContrast = generated.emotion_contrast;
    parameters.maxEmotions = generated.max_emotions;
    parameters.liveBlendCoef = generated.live_blend_coefficient;
    parameters.liveTransitionTime = generated.transition_smoothing;
    parameters.enablePreferredEmotion =
        interactive_emotion_driver_.preferred.has_value();
    if (interactive_emotion_driver_.preferred) {
      const PreferredEmotionSettings& preferred =
          *interactive_emotion_driver_.preferred;
      parameters.preferredEmotionStrength = preferred.strength;
      parameters.preferredEmotion = nva2x::HostTensorFloatConstView(
          preferred.values.data(), preferred.values.size());
    } else {
      parameters.preferredEmotionStrength = 0.0F;
    }
    sdk_check(nva2e::SetInteractiveExecutorPostProcessParameters(
                  *interactive_emotion_executor_, parameters),
              "Configuring interactive Audio2Emotion post-processing");
  }

  void generate_interactive_emotions(std::atomic_bool& canceled) {
    sdk_check(interactive_emotion_accumulator_->Reset(),
              "Resetting interactive emotion accumulator");
    configure_interactive_generated_emotion();

    InteractiveEmotionCapture capture;
    capture.canceled = &canceled;
    capture.accumulator = interactive_emotion_accumulator_.get();
    capture.emotion_count = emotion_channels_.size();
    active_interactive_emotion_capture_ = &capture;
    std::error_code compute_error;
    try {
      compute_error = compute_interactive(
          *interactive_emotion_executor_, canceled);
      sdk_check(cuda_stream_->Synchronize(),
                "Synchronizing interactive emotion results", "gpu_error");
    } catch (...) {
      active_interactive_emotion_capture_ = nullptr;
      throw;
    }
    active_interactive_emotion_capture_ = nullptr;
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Operation was stopped");
    }
    if (capture.failure != nullptr) {
      json details = json::object();
      if (capture.accumulation_error) {
        details = {{"sdk_error", capture.accumulation_error.message()},
                   {"sdk_error_value", capture.accumulation_error.value()}};
      }
      throw WorkerError("inference_failed", capture.failure,
                        std::move(details));
    }
    sdk_check(compute_error, "Computing interactive Audio2Emotion",
              "inference_failed");
    sdk_check(interactive_emotion_accumulator_->Close(),
              "Closing generated interactive emotions");
    sdk_check(interactive_executor_->Invalidate(
                  nva2f::IGeometryInteractiveExecutor::kLayerEmotionAccumulator),
              "Invalidating interactive Audio2Face emotions");

    interactive_effective_emotions_.clear();
    for (auto& [timestamp, host] : capture.frames) {
      std::vector<float> values;
      values.reserve(host->Size());
      for (std::size_t index = 0; index < host->Size(); ++index) {
        const float value = host->Data()[index];
        if (!std::isfinite(value)) {
          throw WorkerError(
              "inference_failed",
              "Interactive Audio2Emotion produced a non-finite value",
              {{"timestamp", timestamp}, {"channel", index}});
        }
        values.push_back(value);
      }
      interactive_effective_emotions_.emplace(timestamp,
                                               std::move(values));
    }
    if (interactive_effective_emotions_.empty()) {
      throw WorkerError("inference_failed",
                        "Interactive Audio2Emotion produced no frames");
    }
    interactive_emotions_valid_ = true;
  }

  static std::vector<float> interpolate_values(
      const std::vector<float>& before, const std::vector<float>& after,
      double factor) {
    if (before.size() != after.size()) {
      throw WorkerError("inference_failed",
                        "Interactive frame widths do not match");
    }
    std::vector<float> result;
    result.reserve(before.size());
    for (std::size_t index = 0; index < before.size(); ++index) {
      const double value = static_cast<double>(before[index]) +
                           (static_cast<double>(after[index]) - before[index]) *
                               factor;
      if (!std::isfinite(value)) {
        throw WorkerError("inference_failed",
                          "Interactive interpolation produced a non-finite value");
      }
      result.push_back(static_cast<float>(value));
    }
    return result;
  }

  std::vector<float> effective_emotions_at(std::int64_t timestamp) const {
    if (interactive_effective_emotions_.empty()) {
      throw WorkerError("inference_failed",
                        "Interactive effective emotion curve is empty");
    }
    auto after = interactive_effective_emotions_.lower_bound(timestamp);
    if (after == interactive_effective_emotions_.begin()) {
      return after->second;
    }
    if (after == interactive_effective_emotions_.end()) {
      return std::prev(after)->second;
    }
    if (after->first == timestamp) return after->second;
    const auto before = std::prev(after);
    const double factor =
        static_cast<double>(timestamp - before->first) /
        static_cast<double>(after->first - before->first);
    return interpolate_values(before->second, after->second, factor);
  }

  void prepare_interactive_settings(const json& settings,
                                    std::atomic_bool& canceled) {
    InferenceSettings parsed = parse_settings(settings);
    configure_interactive_audio2face(parsed.audio2face);
    const json& emotion_settings = settings.at("emotion_driver");
    if (interactive_emotions_valid_ &&
        emotion_settings == interactive_emotion_settings_) {
      return;
    }
    interactive_emotion_driver_ = std::move(parsed.emotion_driver);
    if (interactive_emotion_driver_.generated) {
      generate_interactive_emotions(canceled);
    } else {
      install_interactive_driver_emotions();
    }
    interactive_emotion_settings_ = emotion_settings;
  }

  StreamFrame compute_interactive_frame(std::size_t frame_index,
                                        std::atomic_bool& canceled) {
    Capture capture;
    capture.canceled = &canceled;
    capture.weight_count = interactive_executor_->GetWeightCount();
    active_capture_ = &capture;
    std::error_code compute_error;
    try {
      compute_error = compute_interactive(*interactive_executor_, canceled,
                                          frame_index);
      sdk_check(cuda_stream_->Synchronize(),
                "Synchronizing interactive face results", "gpu_error");
    } catch (...) {
      active_capture_ = nullptr;
      throw;
    }
    active_capture_ = nullptr;
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Operation was stopped");
    }
    if (capture.failure != nullptr) {
      throw WorkerError("inference_failed", capture.failure);
    }
    sdk_check(compute_error, "Computing interactive Audio2Face frame",
              "inference_failed");
    if (capture.frames.size() != 1) {
      throw WorkerError("inference_failed",
                        "Interactive Audio2Face did not return one frame");
    }
    const auto& [timestamp, pending] = *capture.frames.begin();
    if (!pending.weights || !pending.eyes) {
      throw WorkerError("inference_failed",
                        "Interactive SDK callbacks returned an incomplete frame",
                        {{"timestamp", timestamp}});
    }
    std::vector<float> arkit;
    arkit.reserve(pending.weights->Size());
    for (std::size_t channel = 0; channel < pending.weights->Size(); ++channel) {
      const float value = pending.weights->Data()[channel];
      if (!std::isfinite(value)) {
        throw WorkerError(
            "inference_failed",
            "Interactive SDK produced a non-finite blendshape weight",
            {{"timestamp", timestamp}, {"channel", channel}});
      }
      arkit.push_back(value);
    }
    for (std::size_t index = 0; index < pending.eyes->Size(); ++index) {
      if (!std::isfinite(pending.eyes->Data()[index])) {
        throw WorkerError("inference_failed",
                          "Interactive SDK produced a non-finite eye rotation",
                          {{"timestamp", timestamp}, {"component", index}});
      }
    }
    resolve_arkit_eye_look(arkit, pending.eyes->Data(), eye_look_indices_);
    for (float& value : arkit) value = std::clamp(value, 0.0F, 1.0F);
    return StreamFrame{timestamp, std::move(arkit),
                       effective_emotions_at(timestamp)};
  }

  void evaluate_interactive_stream(bool end_of_stream,
                                   std::atomic_bool& canceled,
                                   const StreamFrameCallback& frame_callback) {
    if (retained_audio_.empty()) return;
    const std::int64_t retained_start =
        total_audio_samples_ -
        static_cast<std::int64_t>(retained_audio_.size());
    const std::int64_t alignment =
        static_cast<std::int64_t>(frame_alignment_samples_);
    const std::int64_t offset =
        ((retained_start + alignment - 1) / alignment) * alignment;
    const std::size_t drop =
        static_cast<std::size_t>(offset - retained_start);
    const std::vector<float> window(std::next(retained_audio_.begin(), drop),
                                    retained_audio_.end());
    refill_interactive_audio(window);
    prepare_interactive_settings(interactive_stream_settings_, canceled);
    if (interactive_executor_->GetTotalNbFrames() == 0) return;
    const std::int64_t safe_through =
        end_of_stream
            ? total_audio_samples_
            : total_audio_samples_ - static_cast<std::int64_t>(prebuffer_samples_);
    const std::size_t frame_count = interactive_executor_->GetTotalNbFrames();
    for (std::size_t index = 0; index < frame_count; ++index) {
      const std::int64_t local_timestamp =
          interactive_executor_->GetFrameTimestamp(index);
      if (local_timestamp >
          std::numeric_limits<std::int64_t>::max() - offset) {
        throw WorkerError("inference_failed",
                          "Interactive frame timestamp exceeds the stream timeline");
      }
      const std::int64_t timestamp = offset + local_timestamp;
      if ((previous_timestamp_ && timestamp <= *previous_timestamp_) ||
          timestamp > safe_through) {
        continue;
      }
      if (canceled.load(std::memory_order_acquire)) {
        throw WorkerError("canceled", "Stream was stopped");
      }
      StreamFrame local = compute_interactive_frame(index, canceled);
      local.timestamp_sample = timestamp;
      previous_timestamp_ = timestamp;
      frame_callback(local);
    }
  }

  BakeFrame compute_bake_frame(const BakeFrameRequest& request,
                               std::atomic_bool& canceled) {
    const std::size_t frame_count = interactive_executor_->GetTotalNbFrames();
    std::size_t after_index = 0;
    while (after_index < frame_count &&
           interactive_executor_->GetFrameTimestamp(after_index) <
               request.target_sample) {
      ++after_index;
    }
    if (after_index == frame_count) after_index = frame_count - 1;
    std::size_t before_index = after_index;
    if (interactive_executor_->GetFrameTimestamp(after_index) >
            request.target_sample &&
        after_index > 0) {
      before_index = after_index - 1;
    }
    StreamFrame before = compute_interactive_frame(before_index, canceled);
    StreamFrame after = before_index == after_index
                            ? before
                            : compute_interactive_frame(after_index, canceled);
    std::vector<float> weights = before.weights;
    if (before_index != after_index) {
      const double factor =
          static_cast<double>(request.target_sample -
                              before.timestamp_sample) /
          static_cast<double>(after.timestamp_sample - before.timestamp_sample);
      weights = interpolate_values(before.weights, after.weights, factor);
      for (float& value : weights) value = std::clamp(value, 0.0F, 1.0F);
    }
    return BakeFrame{std::move(weights)};
  }

  void begin_operation(std::uint32_t sample_rate, const json& settings) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    require_model_locked();
    if (sample_rate != sample_rate_) {
      throw WorkerError("sample_rate_mismatch",
                        "Streaming PCM must use the model sample rate",
                        {{"expected", sample_rate_},
                         {"received", sample_rate}});
    }
    if (operation_active_.exchange(true, std::memory_order_acq_rel)) {
      throw WorkerError("busy", "An Audio2Face operation is already active");
    }
    try {
      if (frame_alignment_samples_ == 0 ||
          prebuffer_samples_ >
              std::numeric_limits<std::size_t>::max() - sample_rate_ ||
          prebuffer_samples_ >
              static_cast<std::size_t>(
                  std::numeric_limits<std::int64_t>::max()) - sample_rate_) {
        throw WorkerError("model_invalid",
                          "Model replay context length is out of range");
      }
      const std::size_t base_capacity = prebuffer_samples_ + sample_rate_;
      if (frame_alignment_samples_ - 1 >
          std::numeric_limits<std::size_t>::max() - base_capacity) {
        throw WorkerError("model_invalid",
                          "Model frame alignment is out of range");
      }
      retained_audio_capacity_ =
          base_capacity + frame_alignment_samples_ - 1;
      retained_audio_.clear();
      total_audio_samples_ = 0;
      previous_timestamp_.reset();
      operation_kind_ = OperationKind::Stream;
      InferenceSettings parsed = parse_settings(settings);
      configure_interactive_audio2face(parsed.audio2face);
      interactive_emotion_driver_ = std::move(parsed.emotion_driver);
      if (interactive_emotion_driver_.generated) {
        configure_interactive_generated_emotion();
      }
      interactive_stream_settings_ = settings;
      interactive_emotions_valid_ = false;
      interactive_emotion_settings_ = json();
    } catch (...) {
      operation_kind_ = OperationKind::None;
      operation_active_.store(false, std::memory_order_release);
      throw;
    }
  }

  void finish_operation() noexcept {
    retained_audio_.clear();
    retained_audio_capacity_ = 0;
    total_audio_samples_ = 0;
    previous_timestamp_.reset();
    interactive_stream_settings_ = json();
    bake_audio_.clear();
    bake_prepared_ = false;
    operation_kind_ = OperationKind::None;
    operation_active_.store(false, std::memory_order_release);
  }

  void require_active_stream() const {
    if (!operation_active_.load(std::memory_order_acquire) ||
        operation_kind_ != OperationKind::Stream) {
      throw WorkerError("operation_not_found", "No stream is active");
    }
  }

  void require_active_bake() const {
    if (!operation_active_.load(std::memory_order_acquire) ||
        operation_kind_ != OperationKind::Bake) {
      throw WorkerError("operation_not_found", "No bake is active");
    }
  }

  void retain_audio(const std::vector<float>& audio) {
    if (audio.size() >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max() - total_audio_samples_)) {
      throw WorkerError("invalid_state", "Stream sample timeline is too long");
    }
    retained_audio_.insert(retained_audio_.end(), audio.begin(), audio.end());
    total_audio_samples_ += static_cast<std::int64_t>(audio.size());
    while (retained_audio_.size() > retained_audio_capacity_) {
      retained_audio_.pop_front();
    }
  }

  Audio2FaceSettings parse_audio2face_settings(const json& value) const {
    if (!value.is_object()) {
      throw WorkerError("invalid_params",
                        "settings.audio2face must be an object");
    }
    require_exact_keys(
        value,
        {"input_strength", "lower_face_smoothing", "upper_face_smoothing",
         "lower_face_strength", "upper_face_strength", "face_mask_level",
         "face_mask_softness", "skin_strength", "blink_strength",
         "eyelid_open_offset", "lip_open_offset", "eyeballs_strength",
         "saccade_strength", "right_eye_rot_x_offset",
         "right_eye_rot_y_offset", "left_eye_rot_x_offset",
         "left_eye_rot_y_offset", "eye_saccade_seed"},
        "settings.audio2face");

    Audio2FaceSettings parsed = audio2face_defaults_;
    const auto required_float = [&value](const char* name) {
      return required_float_value(
          value.at(name), std::string("settings.audio2face.") + name);
    };
    parsed.input_strength = required_float("input_strength");
    parsed.skin.lowerFaceSmoothing = required_float("lower_face_smoothing");
    parsed.skin.upperFaceSmoothing = required_float("upper_face_smoothing");
    parsed.skin.lowerFaceStrength = required_float("lower_face_strength");
    parsed.skin.upperFaceStrength = required_float("upper_face_strength");
    parsed.skin.faceMaskLevel = required_float("face_mask_level");
    parsed.skin.faceMaskSoftness = required_float("face_mask_softness");
    parsed.skin.skinStrength = required_float("skin_strength");
    parsed.skin.blinkStrength = required_float("blink_strength");
    parsed.skin.eyelidOpenOffset = required_float("eyelid_open_offset");
    parsed.skin.lipOpenOffset = required_float("lip_open_offset");
    parsed.eyes.eyeballsStrength = required_float("eyeballs_strength");
    parsed.eyes.saccadeStrength = required_float("saccade_strength");
    parsed.eyes.rightEyeballRotationOffsetX =
        required_float("right_eye_rot_x_offset");
    parsed.eyes.rightEyeballRotationOffsetY =
        required_float("right_eye_rot_y_offset");
    parsed.eyes.leftEyeballRotationOffsetX =
        required_float("left_eye_rot_x_offset");
    parsed.eyes.leftEyeballRotationOffsetY =
        required_float("left_eye_rot_y_offset");
    parsed.eyes.saccadeSeed = static_cast<float>(required_size_in_range(
        value, "eye_saccade_seed", "settings.audio2face.", 0, 4999));
    validate_audio2face_settings(parsed, "invalid_params",
                                 "settings.audio2face.");
    return parsed;
  }

  EmotionDriver parse_emotion_driver(const json& value) const {
    if (!value.is_object()) {
      throw WorkerError("invalid_params",
                        "settings.emotion_driver must be an object");
    }
    require_exact_keys(
        value, {"emotion_strength", "generated", "preferred"},
        "settings.emotion_driver");

    EmotionDriver parsed;
    parsed.emotion_strength = required_float_in_range(
        value, "emotion_strength", "settings.emotion_driver.", 0.0F, 2.0F);

    const json& generated = value.at("generated");
    if (!generated.is_null()) {
      if (!generated.is_object()) {
        throw WorkerError(
            "invalid_params",
            "settings.emotion_driver.generated must be null or an object");
      }
      require_exact_keys(
          generated,
          {"emotion_contrast", "max_emotions", "live_blend_coef",
           "transition_smoothing"},
          "settings.emotion_driver.generated");
      GeneratedEmotionSettings generated_settings;
      generated_settings.emotion_contrast = required_float_in_range(
          generated, "emotion_contrast", "settings.emotion_driver.generated.",
          0.1F, 3.0F);
      generated_settings.max_emotions = required_size_in_range(
          generated, "max_emotions", "settings.emotion_driver.generated.", 1,
          emotion_channels_.size());
      generated_settings.live_blend_coefficient = required_float_in_range(
          generated, "live_blend_coef", "settings.emotion_driver.generated.",
          0.0F, 1.0F);
      generated_settings.transition_smoothing = required_float_in_range(
          generated, "transition_smoothing",
          "settings.emotion_driver.generated.", 0.1F, 1.0F);
      parsed.generated = std::move(generated_settings);
    }

    const json& preferred = value.at("preferred");
    if (!preferred.is_null()) {
      if (!preferred.is_object()) {
        throw WorkerError(
            "invalid_params",
            "settings.emotion_driver.preferred must be null or an object");
      }
      require_exact_keys(preferred, {"values", "strength"},
                         "settings.emotion_driver.preferred");
      PreferredEmotionSettings preferred_settings;
      preferred_settings.values = parse_emotion_snapshot(
          preferred.at("values"),
          "settings.emotion_driver.preferred.values");
      preferred_settings.strength = required_float_in_range(
          preferred, "strength", "settings.emotion_driver.preferred.", 0.0F,
          1.0F);
      parsed.preferred = std::move(preferred_settings);
    }
    return parsed;
  }

  InferenceSettings parse_settings(const json& settings) const {
    if (!settings.is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    require_exact_keys(settings, {"audio2face", "emotion_driver"},
                       "settings");
    return {parse_audio2face_settings(settings.at("audio2face")),
            parse_emotion_driver(settings.at("emotion_driver"))};
  }

  std::vector<float> parse_emotion_snapshot(const json& value,
                                            const char* path) const {
    if (!value.is_object() || value.size() != emotion_channels_.size()) {
      throw WorkerError(
          "invalid_params",
          std::string(path) +
              " must contain exactly every model emotion channel");
    }
    std::vector<float> emotion;
    emotion.reserve(emotion_channels_.size());
    for (const std::string& name : emotion_channels_) {
      if (!value.contains(name)) {
        throw WorkerError(
            "invalid_params",
            std::string(path) + " is missing a model emotion channel",
            {{"emotion", name}});
      }
      const float amount = required_float_value(
          value.at(name), std::string(path) + "[" + json(name).dump() + "]");
      if (amount < 0.0F || amount > 1.0F) {
        throw WorkerError(
            "invalid_params",
            std::string(path) + " values must be between 0 and 1",
            {{"emotion", name}, {"received", amount}});
      }
      emotion.push_back(amount);
    }
    return emotion;
  }

  json load_emotion_channels(
      const nva2f::IDiffusionModel::INetworkInfo& network,
      nva2x::HostTensorFloatConstView defaults) {
    const std::size_t count = network.GetEmotionsCount();
    if (defaults.Size() != count || (count != 0 && defaults.Data() == nullptr)) {
      throw WorkerError(
          "model_invalid", "Audio2Face default emotion vector is invalid",
          {{"emotion_count", count}, {"default_count", defaults.Size()}});
    }
    emotion_channels_.clear();
    emotion_channels_.reserve(count);
    json schema = json::array();
    for (std::size_t index = 0; index < count; ++index) {
      const char* name = network.GetEmotionName(index);
      if (name == nullptr || *name == '\0') {
        throw WorkerError("model_invalid",
                          "Audio2Face contains an empty emotion name",
                          {{"index", index}});
      }
      if (std::find(emotion_channels_.begin(), emotion_channels_.end(), name) !=
          emotion_channels_.end()) {
        throw WorkerError("model_invalid",
                          "Audio2Face contains duplicate emotion names",
                          {{"emotion", name}});
      }
      const float value = defaults.Data()[index];
      if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
        throw WorkerError(
            "model_invalid", "Audio2Face default emotion is outside [0, 1]",
            {{"index", index}, {"value", value}});
      }
      emotion_channels_.emplace_back(name);
      schema.push_back({{"name", name}, {"default", value}});
    }
    return schema;
  }

  void clear_locked() noexcept {
    if (cuda_stream_ != nullptr) (void)cuda_stream_->Synchronize();
    active_capture_ = nullptr;
    active_interactive_emotion_capture_ = nullptr;
    interactive_emotion_executor_.reset();
    interactive_executor_.reset();
    interactive_emotion_accumulator_.reset();
    interactive_audio_accumulator_.reset();
    cuda_stream_.reset();
    emotion_model_info_.reset();
    blendshape_model_info_.reset();
    geometry_model_info_.reset();
    eye_look_indices_ = {};
    audio2face_defaults_ = {};
    emotion_channels_.clear();
    finish_operation();
    sample_rate_ = 0;
    prebuffer_samples_ = 0;
    frame_rate_numerator_ = 0;
    frame_rate_denominator_ = 0;
    frame_alignment_samples_ = 0;
    interactive_effective_emotions_.clear();
    interactive_emotion_settings_ = json();
    interactive_emotions_valid_ = false;
  }

  std::mutex resource_mutex_;
  std::mutex interactive_compute_mutex_;
  nva2x::IInteractiveExecutor* active_interactive_compute_{nullptr};
  std::atomic_bool operation_active_{false};
  OperationKind operation_kind_{OperationKind::None};
  SdkPtr<nva2f::IDiffusionModel::IGeometryModelInfo> geometry_model_info_;
  SdkPtr<nva2f::IDiffusionModel::IBlendshapeSolveModelInfo>
      blendshape_model_info_;
  SdkPtr<nva2e::IClassifierModel::IEmotionModelInfo> emotion_model_info_;
  SdkPtr<nva2x::ICudaStream> cuda_stream_;
  SdkPtr<nva2x::IAudioAccumulator> interactive_audio_accumulator_;
  SdkPtr<nva2x::IEmotionAccumulator> interactive_emotion_accumulator_;
  SdkPtr<nva2f::IBlendshapeInteractiveExecutor> interactive_executor_;
  SdkPtr<nva2e::IEmotionInteractiveExecutor> interactive_emotion_executor_;
  Capture* active_capture_{nullptr};
  InteractiveEmotionCapture* active_interactive_emotion_capture_{nullptr};
  ArkitEyeLookIndices eye_look_indices_{};
  Audio2FaceSettings audio2face_defaults_;
  std::vector<std::string> emotion_channels_;
  std::deque<float> retained_audio_;
  std::size_t retained_audio_capacity_{0};
  std::int64_t total_audio_samples_{0};
  std::optional<std::int64_t> previous_timestamp_;
  json interactive_stream_settings_;
  std::vector<float> bake_audio_;
  bool bake_prepared_{false};
  EmotionDriver interactive_emotion_driver_;
  std::map<std::int64_t, std::vector<float>>
      interactive_effective_emotions_;
  json interactive_emotion_settings_;
  bool interactive_emotions_valid_{false};
  std::uint32_t sample_rate_{0};
  std::size_t prebuffer_samples_{0};
  std::size_t frame_rate_numerator_{0};
  std::size_t frame_rate_denominator_{0};
  std::size_t frame_alignment_samples_{0};
};

Backend::Backend() : impl_(std::make_unique<Impl>()) {}
Backend::~Backend() = default;

json Backend::load_model(const ModelRequest& request) {
  return impl_->load_model(request);
}

json Backend::stream_start(const StreamRequest& request) {
  return impl_->stream_start(request);
}

void Backend::stream_chunk(const std::vector<float>& audio,
                           std::atomic_bool& canceled,
                           const StreamFrameCallback& frame) {
  impl_->stream_chunk(audio, canceled, frame);
}

void Backend::stream_settings(const json& settings,
                              std::atomic_bool& canceled) {
  impl_->stream_settings(settings, canceled);
}

void Backend::stream_end(std::atomic_bool& canceled,
                         const StreamFrameCallback& frame) {
  impl_->stream_end(canceled, frame);
}

json Backend::bake_start(const BakeRequest& request) {
  return impl_->bake_start(request);
}

json Backend::bake_chunk(const std::vector<float>& audio,
                         std::atomic_bool& canceled) {
  return impl_->bake_chunk(audio, canceled);
}

json Backend::bake_prepare(std::atomic_bool& canceled) {
  return impl_->bake_prepare(canceled);
}

BakeFrame Backend::bake_frame(const BakeFrameRequest& request,
                              std::atomic_bool& canceled) {
  return impl_->bake_frame(request, canceled);
}

void Backend::bake_end() { impl_->bake_end(); }

void Backend::interrupt_operation() noexcept {
  impl_->interrupt_operation();
}

void Backend::stream_abort() noexcept { impl_->stream_abort(); }

}  // namespace a2f_worker
