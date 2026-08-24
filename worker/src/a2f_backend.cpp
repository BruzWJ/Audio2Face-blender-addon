#include "backend.h"

#include "path_contract.h"

#include <audio2emotion/audio2emotion.h>
#include <audio2face/audio2face.h>
#include <audio2x/cuda_utils.h>
#include <audio2x/executor.h>
#include <audio2x/tensor_float.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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

struct AutomaticEmotionSettings {
  float emotion_strength{};
  float emotion_contrast{};
  std::size_t max_emotions{};
  float live_blend_coefficient{};
  float transition_smoothing{};
  std::optional<std::vector<float>> preferred_emotion;
  float preferred_emotion_strength{};
};

struct Audio2FaceSettings {
  float input_strength{0.0F};
  nva2f::AnimatorSkinParams skin{};
  nva2f::AnimatorEyesParams eyes{};
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

bool required_bool(const json& object, const char* name, const char* path) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_boolean()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " must be boolean");
  }
  return it->get<bool>();
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
      (void)require_canonical_regular_file(
          request.audio2face_model_path, "model_not_found",
          "audio2face_model_path");
      (void)require_canonical_regular_file(
          request.audio2emotion_model_path, "model_not_found",
          "audio2emotion_model_path");
      sdk_check(nva2x::SetCudaDeviceIfNeeded(0), "Selecting CUDA device",
                "gpu_error");
      const auto geometry_info = require_sdk_ptr(
          nva2f::ReadDiffusionModelInfo(
              request.audio2face_model_path.c_str()),
          "Reading diffusion model", "model_invalid");
      const auto& network = geometry_info->GetNetworkInfo();
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
      if (network.GetIdentityLength() == 0) {
        throw WorkerError("model_invalid",
                          "Audio2Face model has no identity");
      }
      json emotion_channels =
          load_emotion_channels(network, network.GetDefaultEmotion());

      const auto blendshape_info = require_sdk_ptr(
          nva2f::ReadDiffusionBlendshapeSolveModelInfo(
              request.audio2face_model_path.c_str()),
          "Reading diffusion blendshape solver", "model_invalid");
      const auto execution_option =
          nva2f::IGeometryExecutor::ExecutionOption::Skin |
          nva2f::IGeometryExecutor::ExecutionOption::Eyes;
      const auto blendshape_parameters =
          blendshape_info->GetExecutorCreationParameters(
              execution_option, kDefaultIdentityIndex);
      std::vector<std::string> output_channels =
          skin_pose_names(blendshape_parameters.initializationSkinParams);
      validate_arkit52_channels(output_channels);
      eye_look_indices_ = resolve_arkit_eye_look_indices(output_channels);

      bundle_ = require_sdk_ptr(
          nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
              1, request.audio2face_model_path.c_str(),
              execution_option, true, kDefaultIdentityIndex, true, nullptr,
              nullptr),
          "Creating diffusion GPU blendshape executor", "gpu_error");
      auto& executor = bundle_->GetExecutor();
      if (executor.GetResultType() !=
          nva2f::IBlendshapeExecutor::ResultsType::DEVICE) {
        throw WorkerError("gpu_error",
                          "Audio2Face did not create a device blendshape solver");
      }
      if (executor.GetNbTracks() != 1 ||
          executor.GetWeightCount() != output_channels.size()) {
        throw WorkerError(
            "model_invalid", "SDK blendshape layout does not match model metadata",
            {{"tracks", executor.GetNbTracks()},
             {"reported_weights", executor.GetWeightCount()},
             {"expected_weights", output_channels.size()}});
      }
      auto& geometry = geometry_executor();
      // BlendshapeSolveExecutor::Init() in SDK 1.0.0 reapplies an execution
      // option derived only from its skin/tongue solvers. Restore the eye
      // output requested above before installing the callbacks.
      sdk_check(geometry.SetExecutionOption(execution_option),
                "Enabling skin and eye geometry outputs");
      if (geometry.GetExecutionOption() != execution_option) {
        throw WorkerError(
            "sdk_error", "Geometry executor did not enable skin and eye outputs");
      }
      if (geometry.GetEyesRotationSize() != kEyesRotationCount) {
        throw WorkerError("model_invalid", "Unsupported SDK eyes rotation size",
                          {{"reported", geometry.GetEyesRotationSize()},
                           {"expected", kEyesRotationCount}});
      }
      sdk_check(nva2f::SetExecutorGeometryResultsCallback(
                    executor, &Impl::geometry_callback, this),
                "Installing geometry callback");
      sdk_check(executor.SetResultsCallback(&Impl::weights_callback, this),
                "Installing device blendshape callback");

      Audio2FaceSettings audio2face_defaults;
      sdk_check(nva2f::GetExecutorInputStrength(
                    geometry, audio2face_defaults.input_strength),
                "Reading Audio2Face input strength");
      sdk_check(nva2f::GetExecutorSkinParameters(
                    geometry, 0, audio2face_defaults.skin),
                "Reading Audio2Face skin parameters");
      sdk_check(nva2f::GetExecutorEyesParameters(
                    geometry, 0, audio2face_defaults.eyes),
                "Reading Audio2Face eyes parameters");
      validate_audio2face_settings(audio2face_defaults, "model_invalid",
                                   "Audio2Face model tuning default ");
      audio2face_defaults_ = audio2face_defaults;

      const std::size_t sample_rate = executor.GetSamplingRate();
      if (sample_rate == 0 || sample_rate > kMaximumSupportedSampleRate) {
        throw WorkerError("model_invalid", "SDK reported an invalid sample rate",
                          {{"sample_rate", sample_rate}});
      }
      sample_rate_ = static_cast<std::uint32_t>(sample_rate);
      if (audio2face_network.bufferSamplerate != sample_rate_) {
        throw WorkerError(
            "model_invalid",
            "Audio2Face model and executor sample rates do not match",
            {{"model_sample_rate", audio2face_network.bufferSamplerate},
             {"executor_sample_rate", sample_rate_}});
      }

      const auto emotion_model_info = require_sdk_ptr(
          nva2e::ReadClassifierModelInfo(
              request.audio2emotion_model_path.c_str()),
          "Reading Audio2Emotion classifier model", "model_invalid");
      std::size_t frame_rate_numerator = 0;
      std::size_t frame_rate_denominator = 0;
      executor.GetFrameRate(frame_rate_numerator, frame_rate_denominator);
      if (frame_rate_numerator == 0 || frame_rate_denominator == 0) {
        throw WorkerError("model_invalid",
                          "Audio2Face reported an invalid frame rate");
      }
      nva2e::EmotionExecutorCreationParameters emotion_parameters;
      emotion_parameters.cudaStream = bundle_->GetCudaStream().Data();
      emotion_parameters.nbTracks = 1;
      const nva2x::IAudioAccumulator* shared_audio_accumulator =
          &bundle_->GetAudioAccumulator(0);
      emotion_parameters.sharedAudioAccumulators = &shared_audio_accumulator;
      const auto classifier_parameters =
          emotion_model_info->GetExecutorCreationParameters(
              kAudio2EmotionInputWindowSamples, frame_rate_numerator,
              frame_rate_denominator, kAudio2EmotionInferencesToSkip);
      const std::size_t audio2emotion_input_window_samples =
          classifier_parameters.networkInfo.bufferLength;
      if (audio2emotion_input_window_samples == 0 ||
          classifier_parameters.networkInfo.bufferSamplerate != sample_rate_) {
        throw WorkerError(
            "model_invalid",
            "Audio2Emotion reported invalid audio window metadata",
            {{"buffer_samples", audio2emotion_input_window_samples},
             {"audio2emotion_sample_rate",
              classifier_parameters.networkInfo.bufferSamplerate},
             {"audio2face_sample_rate", sample_rate_}});
      }
      prebuffer_samples_ = std::max(audio2face_input_window_samples,
                                    audio2emotion_input_window_samples);
      emotion_executor_ = require_sdk_ptr(
          nva2e::CreateClassifierEmotionExecutor(emotion_parameters,
                                                 classifier_parameters),
          "Creating Audio2Emotion GPU executor", "gpu_error");
      // SDK 1.0.0 exposes the post-processed vector width, but not names for
      // those output positions. Require the selected v3.0 model pair to expose
      // the same ordered emotion-vector width before creating the integration.
      if (emotion_executor_->GetNbTracks() != 1 ||
          emotion_executor_->GetSamplingRate() != sample_rate_ ||
          emotion_executor_->GetEmotionsSize() != emotion_channels_.size()) {
        throw WorkerError(
            "model_invalid",
            "Audio2Emotion output is incompatible with Audio2Face",
            {{"audio2face_sample_rate", sample_rate_},
             {"audio2emotion_sample_rate",
              emotion_executor_->GetSamplingRate()},
             {"audio2face_emotion_count", emotion_channels_.size()},
             {"audio2emotion_output_count",
              emotion_executor_->GetEmotionsSize()}});
      }
      sdk_check(emotion_executor_->SetResultsCallback(
                    &Impl::emotion_callback, this),
                "Installing Audio2Emotion callback");

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
    accumulate_audio(audio.data(), audio.size());
    retain_audio(audio);
    drain_ready(canceled, frame);
  }

  void stream_settings(const json& settings,
                       std::atomic_bool& canceled,
                       const StreamResetCallback& reset,
                       const StreamFrameCallback& frame) {
    require_active_stream();
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    reset_inference(settings);
    timestamp_offset_ =
        total_audio_samples_ - static_cast<std::int64_t>(retained_audio_.size());
    previous_timestamp_.reset();
    reset();
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    if (!retained_audio_.empty()) {
      const std::vector<float> replay(retained_audio_.begin(),
                                      retained_audio_.end());
      accumulate_audio(replay.data(), replay.size());
      drain_ready(canceled, frame);
    }
  }

  void stream_end(std::atomic_bool& canceled,
                  const StreamFrameCallback& frame) {
    require_active_stream();
    OperationReset reset(*this);
    close_audio_and_drain(canceled, frame);
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

  struct EmotionCapture {
    std::atomic_bool* canceled{nullptr};
    nva2x::IEmotionAccumulator* accumulator{nullptr};
    std::size_t emotion_count{0};
    std::error_code accumulation_error;
    const char* failure{nullptr};
  };

  static void fail_capture(Capture& capture, const char* message) noexcept {
    capture.failure = message;
  }

  static bool emotion_callback(
      void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    auto& owner = *static_cast<Impl*>(userdata);
    if (owner.active_emotion_capture_ == nullptr) return false;
    auto& capture = *owner.active_emotion_capture_;
    if (capture.canceled->load(std::memory_order_acquire)) return false;
    if (results.trackIndex != 0) {
      capture.failure = "Audio2Emotion callback returned an unexpected track";
      return false;
    }
    if (results.emotions.Size() != capture.emotion_count) {
      capture.failure = "Audio2Emotion callback returned an unexpected value count";
      return false;
    }
    const std::error_code error = capture.accumulator->Accumulate(
        results.timeStampCurrentFrame, results.emotions, results.cudaStream);
    if (error) {
      capture.accumulation_error = error;
      capture.failure = "Accumulating generated emotion failed";
      return false;
    }
    return true;
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
          (frame.weights &&
           frame.next_timestamp != results.timeStampNextFrame)) {
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
          (frame.eyes && frame.next_timestamp != results.timeStampNextFrame)) {
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

  nva2f::IBlendshapeExecutor& executor() { return bundle_->GetExecutor(); }

  nva2f::IGeometryExecutor& geometry_executor() {
    nva2f::IGeometryExecutor* geometry = nullptr;
    sdk_check(nva2f::GetExecutorGeometryExecutor(executor(), &geometry),
              "Retrieving geometry executor");
    if (geometry == nullptr) {
      throw WorkerError("sdk_error", "Geometry executor is null");
    }
    return *geometry;
  }

  void require_model_locked() const {
    if (bundle_ == nullptr) {
      throw WorkerError("model_not_loaded", "Load a model before inference");
    }
  }

  void reset_inference(const json& settings) {
    sdk_check(executor().Wait(0), "Waiting for prior blendshape work",
              "gpu_error");
    sdk_check(bundle_->GetCudaStream().Synchronize(),
              "Synchronizing CUDA stream", "gpu_error");
    sdk_check(executor().Reset(0), "Resetting blendshape executor");
    sdk_check(bundle_->GetAudioAccumulator(0).Reset(),
              "Resetting audio accumulator");
    sdk_check(bundle_->GetEmotionAccumulator(0).Reset(),
              "Resetting emotion accumulator");
    sdk_check(emotion_executor_->Reset(0),
              "Resetting Audio2Emotion executor");
    apply_settings(settings);
    if (auto_audio2emotion_active_) {
      configure_automatic_emotion();
    } else {
      sdk_check(bundle_->GetEmotionAccumulator(0).Accumulate(
                    0,
                    nva2x::HostTensorFloatConstView(
                        manual_emotion_.data(), manual_emotion_.size()),
                    bundle_->GetCudaStream().Data()),
                "Accumulating manual emotion driver");
      sdk_check(bundle_->GetEmotionAccumulator(0).Close(),
                "Closing manual emotion driver");
    }
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
      if (prebuffer_samples_ >
              std::numeric_limits<std::size_t>::max() - sample_rate_ ||
          prebuffer_samples_ >
              static_cast<std::size_t>(
                  std::numeric_limits<std::int64_t>::max()) - sample_rate_) {
        throw WorkerError("model_invalid",
                          "Model replay context length is out of range");
      }
      retained_audio_capacity_ = prebuffer_samples_ + sample_rate_;
      retained_audio_.clear();
      total_audio_samples_ = 0;
      timestamp_offset_ = 0;
      previous_timestamp_.reset();
      reset_inference(settings);
    } catch (...) {
      operation_active_.store(false, std::memory_order_release);
      throw;
    }
  }

  void finish_operation() noexcept {
    retained_audio_.clear();
    retained_audio_capacity_ = 0;
    total_audio_samples_ = 0;
    timestamp_offset_ = 0;
    previous_timestamp_.reset();
    operation_active_.store(false, std::memory_order_release);
  }

  void require_active_stream() const {
    if (!operation_active_.load(std::memory_order_acquire)) {
      throw WorkerError("operation_not_found", "No stream is active");
    }
  }

  void accumulate_audio(const float* audio, std::size_t count) {
    sdk_check(bundle_->GetAudioAccumulator(0).Accumulate(
                  nva2x::HostTensorFloatConstView(audio, count),
                  bundle_->GetCudaStream().Data()),
              "Accumulating audio");
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

  void close_audio_and_drain(std::atomic_bool& canceled,
                             const StreamFrameCallback& frame) {
    sdk_check(bundle_->GetAudioAccumulator(0).Close(),
              "Closing audio accumulator");
    drain_interleaved_ready(canceled, frame);
    if (auto_audio2emotion_active_) {
      if (emotion_executor_->GetNbAvailableExecutions(0) != 0) {
        throw WorkerError(
            "inference_failed",
            "Audio2Emotion did not consume all available audio");
      }
      sdk_check(bundle_->GetEmotionAccumulator(0).Close(),
                "Closing generated emotion stream");
      drain_interleaved_ready(canceled, frame);
    }
    sdk_check(executor().Wait(0), "Waiting for blendshape results",
              "gpu_error");
    sdk_check(bundle_->GetCudaStream().Synchronize(),
              "Synchronizing CUDA stream", "gpu_error");
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Operation was stopped");
    }
  }

  void drain_ready(std::atomic_bool& canceled,
                   const StreamFrameCallback& frame_callback) {
    drain_interleaved_ready(canceled, frame_callback);
    drop_consumed_inputs();
  }

  void drain_interleaved_ready(
      std::atomic_bool& canceled,
      const StreamFrameCallback& frame_callback) {
    while (true) {
      if (nva2x::GetNbReadyTracks(executor()) > 0) {
        execute_face_once(canceled, frame_callback);
        continue;
      }
      if (auto_audio2emotion_active_ &&
          nva2x::GetNbReadyTracks(*emotion_executor_) > 0) {
        execute_emotion_once(canceled);
        continue;
      }
      break;
    }
  }

  void execute_emotion_once(std::atomic_bool& canceled) {
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Operation was stopped");
    }
    EmotionCapture capture;
    capture.canceled = &canceled;
    capture.accumulator = &bundle_->GetEmotionAccumulator(0);
    capture.emotion_count = emotion_channels_.size();
    active_emotion_capture_ = &capture;
    std::error_code execute_error;
    try {
      execute_error = emotion_executor_->Execute(nullptr);
    } catch (...) {
      active_emotion_capture_ = nullptr;
      throw;
    }
    active_emotion_capture_ = nullptr;
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
    sdk_check(execute_error, "Executing Audio2Emotion", "inference_failed");
  }

  std::int64_t absolute_timestamp(std::int64_t local_timestamp) const {
    if (local_timestamp >
        std::numeric_limits<std::int64_t>::max() - timestamp_offset_) {
      throw WorkerError("inference_failed",
                        "SDK frame timestamp exceeds the stream timeline");
    }
    return timestamp_offset_ + local_timestamp;
  }

  void execute_face_once(std::atomic_bool& canceled,
                         const StreamFrameCallback& frame_callback) {
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Operation was stopped");
    }
    Capture capture;
    capture.canceled = &canceled;
    capture.weight_count = executor().GetWeightCount();
    active_capture_ = &capture;
    std::error_code execute_error;
    try {
      execute_error = executor().Execute(nullptr);
      sdk_check(bundle_->GetCudaStream().Synchronize(),
                "Synchronizing frame results", "gpu_error");
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
    sdk_check(execute_error, "Executing Audio2Face", "inference_failed");

    for (const auto& [timestamp, pending] : capture.frames) {
      const std::int64_t stream_timestamp = absolute_timestamp(timestamp);
      if (!pending.weights || !pending.eyes) {
        throw WorkerError("inference_failed",
                          "SDK callbacks returned an incomplete frame",
                          {{"timestamp", stream_timestamp}});
      }
      if (previous_timestamp_.has_value() &&
          stream_timestamp <= *previous_timestamp_) {
        throw WorkerError("inference_failed",
                          "SDK frame timestamps are not strictly increasing",
                          {{"timestamp", stream_timestamp},
                           {"previous", *previous_timestamp_}});
      }
      std::vector<float> arkit;
      arkit.reserve(pending.weights->Size());
      for (std::size_t channel = 0; channel < pending.weights->Size(); ++channel) {
        const float value = pending.weights->Data()[channel];
        if (!std::isfinite(value)) {
          throw WorkerError(
              "inference_failed",
              "SDK produced a non-finite blendshape weight",
              {{"timestamp", stream_timestamp}, {"channel", channel}});
        }
        arkit.push_back(value);
      }
      for (std::size_t index = 0; index < pending.eyes->Size(); ++index) {
        if (!std::isfinite(pending.eyes->Data()[index])) {
          throw WorkerError("inference_failed",
                            "SDK produced a non-finite eye rotation",
                            {{"timestamp", stream_timestamp},
                             {"component", index}});
        }
      }
      resolve_arkit_eye_look(arkit, pending.eyes->Data(), eye_look_indices_);
      for (float& value : arkit) {
        if (!std::isfinite(value)) {
          throw WorkerError("inference_failed",
                            "ARKit resolver produced a non-finite weight",
                            {{"timestamp", stream_timestamp}});
        }
        value = std::clamp(value, 0.0F, 1.0F);
      }
      if (canceled.load(std::memory_order_acquire)) {
        throw WorkerError("canceled", "Operation was stopped");
      }
      previous_timestamp_ = stream_timestamp;
      frame_callback(StreamFrame{stream_timestamp, std::move(arkit)});
    }
  }

  void drop_consumed_inputs() {
    std::size_t next_audio_sample = executor().GetNextAudioSampleToRead(0);
    if (auto_audio2emotion_active_) {
      next_audio_sample = std::min(
          next_audio_sample, emotion_executor_->GetNextAudioSampleToRead(0));
    }
    sdk_check(bundle_->GetAudioAccumulator(0).DropSamplesBefore(
                  next_audio_sample),
              "Dropping processed audio samples");

    auto& emotion_accumulator = bundle_->GetEmotionAccumulator(0);
    if (auto_audio2emotion_active_ && !emotion_accumulator.IsEmpty()) {
      const auto next_emotion_timestamp =
          executor().GetNextEmotionTimestampToRead(0);
      const auto last_emotion_timestamp =
          emotion_accumulator.LastAccumulatedTimestamp();
      sdk_check(emotion_accumulator.DropEmotionsBefore(
                    std::min(next_emotion_timestamp, last_emotion_timestamp)),
                "Dropping processed emotions");
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

  void configure_audio2face(const Audio2FaceSettings& settings) {
    auto& geometry = geometry_executor();
    sdk_check(nva2f::SetExecutorInputStrength(geometry,
                                              settings.input_strength),
              "Configuring Audio2Face input strength");
    sdk_check(nva2f::SetExecutorSkinParameters(geometry, 0, settings.skin),
              "Configuring Audio2Face skin parameters");
    sdk_check(nva2f::SetExecutorEyesParameters(geometry, 0, settings.eyes),
              "Configuring Audio2Face eyes parameters");
  }

  void apply_settings(const json& settings) {
    if (!settings.is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    require_exact_keys(
        settings,
        {"audio2face", "auto_audio2emotion", "manual_emotions",
         "audio2emotion"},
        "settings");
    const Audio2FaceSettings audio2face =
        parse_audio2face_settings(settings.at("audio2face"));
    const bool auto_audio2emotion = required_bool(
        settings, "auto_audio2emotion", "settings.");
    const auto manual_value = settings.find("manual_emotions");
    std::vector<float> manual_emotion = parse_emotion_snapshot(
        *manual_value, "settings.manual_emotions");

    const auto automatic_value = settings.find("audio2emotion");
    if (automatic_value == settings.end() || !automatic_value->is_object()) {
      throw WorkerError("invalid_params",
                        "settings.audio2emotion must be an object");
    }
    require_exact_keys(
        *automatic_value,
        {"emotion_strength", "emotion_contrast", "max_emotions",
         "live_blend_coef", "transition_smoothing",
         "preferred_emotion", "preferred_emotion_strength"},
        "settings.audio2emotion");
    AutomaticEmotionSettings automatic_emotion;
    automatic_emotion.emotion_strength = required_float_in_range(
        *automatic_value, "emotion_strength", "settings.audio2emotion.", 0.0F,
        1.0F);
    automatic_emotion.emotion_contrast = required_float_in_range(
        *automatic_value, "emotion_contrast", "settings.audio2emotion.", 0.1F,
        3.0F);
    automatic_emotion.max_emotions = required_size_in_range(
        *automatic_value, "max_emotions", "settings.audio2emotion.", 1,
        emotion_channels_.size());
    automatic_emotion.live_blend_coefficient = required_float_in_range(
        *automatic_value, "live_blend_coef", "settings.audio2emotion.", 0.0F,
        1.0F);
    automatic_emotion.transition_smoothing = required_float_in_range(
        *automatic_value, "transition_smoothing", "settings.audio2emotion.",
        0.1F, 1.0F);
    const auto preferred_value = automatic_value->find("preferred_emotion");
    if (!preferred_value->is_null()) {
      automatic_emotion.preferred_emotion = parse_emotion_snapshot(
          *preferred_value, "settings.audio2emotion.preferred_emotion");
    }
    automatic_emotion.preferred_emotion_strength = required_float_in_range(
        *automatic_value, "preferred_emotion_strength",
        "settings.audio2emotion.", 0.0F, 1.0F);

    auto_audio2emotion_active_ = auto_audio2emotion;
    manual_emotion_ = std::move(manual_emotion);
    automatic_emotion_settings_ = std::move(automatic_emotion);
    configure_audio2face(audio2face);
  }

  void configure_automatic_emotion() {
    nva2e::PostProcessParams parameters;
    sdk_check(nva2e::GetExecutorPostProcessParameters(
                  *emotion_executor_, 0, parameters),
              "Reading Audio2Emotion post-process parameters");
    parameters.emotionStrength =
        automatic_emotion_settings_.emotion_strength;
    parameters.emotionContrast =
        automatic_emotion_settings_.emotion_contrast;
    parameters.maxEmotions = automatic_emotion_settings_.max_emotions;
    parameters.liveBlendCoef =
        automatic_emotion_settings_.live_blend_coefficient;
    parameters.liveTransitionTime =
        automatic_emotion_settings_.transition_smoothing;
    parameters.enablePreferredEmotion =
        automatic_emotion_settings_.preferred_emotion.has_value();
    parameters.preferredEmotionStrength =
        automatic_emotion_settings_.preferred_emotion_strength;
    if (automatic_emotion_settings_.preferred_emotion) {
      const std::vector<float>& preferred_emotion =
          *automatic_emotion_settings_.preferred_emotion;
      parameters.preferredEmotion = nva2x::HostTensorFloatConstView(
          preferred_emotion.data(), preferred_emotion.size());
    }
    sdk_check(nva2e::SetExecutorPostProcessParameters(
                  *emotion_executor_, 0, parameters),
              "Configuring Audio2Emotion post-processing");
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
    if (bundle_ != nullptr) {
      (void)bundle_->GetExecutor().Wait(0);
      (void)bundle_->GetCudaStream().Synchronize();
    }
    active_capture_ = nullptr;
    active_emotion_capture_ = nullptr;
    emotion_executor_.reset();
    bundle_.reset();
    eye_look_indices_ = {};
    audio2face_defaults_ = {};
    emotion_channels_.clear();
    manual_emotion_.clear();
    finish_operation();
    sample_rate_ = 0;
    prebuffer_samples_ = 0;
    auto_audio2emotion_active_ = false;
    automatic_emotion_settings_ = {};
  }

  std::mutex resource_mutex_;
  std::atomic_bool operation_active_{false};
  SdkPtr<nva2f::IBlendshapeExecutorBundle> bundle_;
  SdkPtr<nva2e::IEmotionExecutor> emotion_executor_;
  Capture* active_capture_{nullptr};
  EmotionCapture* active_emotion_capture_{nullptr};
  ArkitEyeLookIndices eye_look_indices_{};
  Audio2FaceSettings audio2face_defaults_;
  std::vector<std::string> emotion_channels_;
  std::vector<float> manual_emotion_;
  std::deque<float> retained_audio_;
  std::size_t retained_audio_capacity_{0};
  std::int64_t total_audio_samples_{0};
  std::int64_t timestamp_offset_{0};
  std::optional<std::int64_t> previous_timestamp_;
  std::uint32_t sample_rate_{0};
  std::size_t prebuffer_samples_{0};
  bool auto_audio2emotion_active_{false};
  AutomaticEmotionSettings automatic_emotion_settings_;
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
                              std::atomic_bool& canceled,
                              const StreamResetCallback& reset,
                              const StreamFrameCallback& frame) {
  impl_->stream_settings(settings, canceled, reset, frame);
}

void Backend::stream_end(std::atomic_bool& canceled,
                         const StreamFrameCallback& frame) {
  impl_->stream_end(canceled, frame);
}

void Backend::stream_abort() noexcept { impl_->stream_abort(); }

}  // namespace a2f_worker
