#include "a2f_worker/backend.h"

#include "a2f_worker/result_file.h"
#include "a2f_worker/wav.h"

#include <audio2emotion/audio2emotion.h>
#include <audio2face/audio2face.h>
#include <audio2x/cuda_utils.h>
#include <audio2x/executor.h>
#include <audio2x/tensor_float.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <iterator>
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

constexpr std::size_t kAudioChunkSamples = 16000;
constexpr std::size_t kAudio2EmotionInputWindowSamples = 60000;
constexpr std::size_t kAudio2EmotionInferencesToSkip = 30;
constexpr std::size_t kMaximumResultScalars = 40000000;
constexpr std::size_t kEyesRotationCount = 6;
constexpr std::uint32_t kMaximumSupportedSampleRate = 384000;

// Exact NVIDIA skin names in the order defined by a2f-animation/1.
constexpr std::array<const char*, 52> kArkit52SdkNames = {{
    "eyeBlinkLeft",       "eyeLookDownLeft",    "eyeLookInLeft",
    "eyeLookOutLeft",     "eyeLookUpLeft",      "eyeSquintLeft",
    "eyeWideLeft",        "eyeBlinkRight",      "eyeLookDownRight",
    "eyeLookInRight",     "eyeLookOutRight",    "eyeLookUpRight",
    "eyeSquintRight",     "eyeWideRight",       "jawForward",
    "jawLeft",            "jawRight",           "jawOpen",
    "mouthClose",         "mouthFunnel",        "mouthPucker",
    "mouthLeft",          "mouthRight",         "mouthSmileLeft",
    "mouthSmileRight",    "mouthFrownLeft",     "mouthFrownRight",
    "mouthDimpleLeft",    "mouthDimpleRight",   "mouthStretchLeft",
    "mouthStretchRight",  "mouthRollLower",     "mouthRollUpper",
    "mouthShrugLower",    "mouthShrugUpper",    "mouthPressLeft",
    "mouthPressRight",    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft",   "mouthUpperUpRight",  "browDownLeft",
    "browDownRight",      "browInnerUp",        "browOuterUpLeft",
    "browOuterUpRight",   "cheekPuff",          "cheekSquintLeft",
    "cheekSquintRight",   "noseSneerLeft",      "noseSneerRight",
    "tongueOut",
}};

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

float required_float(const json& object, const char* name, const char* path) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_number()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " must be numeric");
  }
  const double parsed = it->get<double>();
  if (!std::isfinite(parsed) || parsed < -std::numeric_limits<float>::max() ||
      parsed > std::numeric_limits<float>::max()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " must be finite");
  }
  return static_cast<float>(parsed);
}

float required_float_range(const json& object, const char* name,
                           const char* path, float minimum, float maximum) {
  const float value = required_float(object, name, path);
  if (value < minimum || value > maximum) {
    throw WorkerError(
        "invalid_params", std::string(path) + name + " is outside its range",
        {{"minimum", minimum}, {"maximum", maximum}, {"received", value}});
  }
  return value;
}

bool required_bool(const json& object, const char* name, const char* path) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_boolean()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " must be boolean");
  }
  return it->get<bool>();
}

std::size_t required_size_range(const json& object, const char* name,
                                const char* path, std::size_t minimum,
                                std::size_t maximum) {
  const auto it = object.find(name);
  if (it == object.end() || !it->is_number_integer()) {
    throw WorkerError("invalid_params",
                      std::string(path) + name +
                          " must be an unsigned integer");
  }
  const auto signed_value = it->get<std::int64_t>();
  if (signed_value < 0) {
    throw WorkerError("invalid_params",
                      std::string(path) + name + " must not be negative");
  }
  const auto value = static_cast<std::uint64_t>(signed_value);
  if (value < minimum || value > maximum) {
    throw WorkerError(
        "invalid_params", std::string(path) + name + " is outside its range",
        {{"minimum", minimum}, {"maximum", maximum}, {"received", value}});
  }
  return static_cast<std::size_t>(value);
}

json skin_json(const nva2f::AnimatorSkinParams& value) {
  return {{"lower_face_smoothing", value.lowerFaceSmoothing},
          {"upper_face_smoothing", value.upperFaceSmoothing},
          {"lower_face_strength", value.lowerFaceStrength},
          {"upper_face_strength", value.upperFaceStrength},
          {"face_mask_level", value.faceMaskLevel},
          {"face_mask_softness", value.faceMaskSoftness},
          {"skin_strength", value.skinStrength},
          {"blink_strength", value.blinkStrength},
          {"eyelid_open_offset", value.eyelidOpenOffset},
          {"lip_open_offset", value.lipOpenOffset},
          {"blink_offset", value.blinkOffset}};
}

std::vector<std::string> pose_names(
    const nva2f::BlendshapeSolveExecutorCreationParameters::BlendshapeParams* params,
    const char* solver_name) {
  if (params == nullptr) {
    throw WorkerError("model_invalid",
                      std::string("Model has no ") + solver_name +
                          " blendshape solver");
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

std::vector<std::size_t> arkit52_indices(
    const std::vector<std::string>& sdk_names) {
  std::vector<std::size_t> indices;
  indices.reserve(kArkit52SdkNames.size());
  json missing = json::array();
  json ambiguous = json::array();
  json unexpected = json::array();
  for (const char* sdk_name : kArkit52SdkNames) {
    const auto found = std::find(sdk_names.begin(), sdk_names.end(), sdk_name);
    if (found == sdk_names.end()) {
      missing.push_back(sdk_name);
      continue;
    }
    if (std::find(std::next(found), sdk_names.end(), sdk_name) != sdk_names.end()) {
      ambiguous.push_back(sdk_name);
      continue;
    }
    indices.push_back(
        static_cast<std::size_t>(std::distance(sdk_names.begin(), found)));
  }
  for (const std::string& sdk_name : sdk_names) {
    if (std::find(kArkit52SdkNames.begin(), kArkit52SdkNames.end(), sdk_name) ==
        kArkit52SdkNames.end()) {
      unexpected.push_back(sdk_name);
    }
  }
  if (!missing.empty() || !ambiguous.empty() || !unexpected.empty() ||
      sdk_names.size() != kArkit52SdkNames.size()) {
    throw WorkerError(
        "model_invalid",
        "Skin blendshape solver does not match NVIDIA's ARKit-52 contract",
        {{"required_schema", "arkit-52/1"},
         {"missing", missing},
         {"ambiguous", ambiguous},
         {"unexpected", unexpected},
         {"sdk_skin_channel_names", sdk_names}});
  }
  return indices;
}

void resolve_arkit_eye_look(std::vector<float>& weights,
                            const float* eyes) {
  constexpr float kEyeRangeDegrees = 60.0F;
  const float right_x = eyes[0] / kEyeRangeDegrees;
  const float right_y = eyes[1] / kEyeRangeDegrees;
  const float left_x = eyes[3] / kEyeRangeDegrees;
  const float left_y = eyes[4] / kEyeRangeDegrees;
  weights[1] = left_x;
  weights[2] = -left_y;
  weights[3] = left_y;
  weights[4] = -left_x;
  weights[8] = right_x;
  weights[9] = right_y;
  weights[10] = -right_y;
  weights[11] = -right_x;
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
      std::error_code fs_error;
      if (!std::filesystem::is_regular_file(request.audio2face_model_path,
                                            fs_error)) {
        throw WorkerError(
            "model_not_found",
            "audio2face_model_path is not a regular file",
            {{"path", request.audio2face_model_path}});
      }
      fs_error.clear();
      if (!std::filesystem::is_regular_file(request.audio2emotion_model_path,
                                            fs_error)) {
        throw WorkerError(
            "model_not_found",
            "audio2emotion_model_path is not a regular file",
            {{"path", request.audio2emotion_model_path}});
      }
      sdk_check(nva2x::SetCudaDeviceIfNeeded(0), "Selecting CUDA device",
                "gpu_error");
      diffusion_geometry_info_ = require_sdk_ptr(
          nva2f::ReadDiffusionModelInfo(
              request.audio2face_model_path.c_str()),
          "Reading diffusion model", "model_invalid");
      const auto& network = diffusion_geometry_info_->GetNetworkInfo();
      const auto& audio2face_network = network.GetNetworkInfo();
      audio2face_input_window_samples_ = audio2face_network.bufferLength;
      if (audio2face_input_window_samples_ == 0 ||
          audio2face_network.bufferSamplerate == 0) {
        throw WorkerError(
            "model_invalid",
            "Audio2Face reported invalid audio window metadata",
            {{"buffer_samples", audio2face_input_window_samples_},
             {"sample_rate", audio2face_network.bufferSamplerate}});
      }
      if (static_cast<std::size_t>(request.identity_index) >=
          network.GetIdentityLength()) {
        throw WorkerError(
            "identity_invalid",
            "identity_index is outside the model identity range",
            {{"identity_index", request.identity_index},
             {"identity_count", network.GetIdentityLength()}});
      }
      copy_default_emotion(network.GetDefaultEmotion());
      if (default_emotion_.size() != network.GetEmotionsCount()) {
        throw WorkerError("model_invalid",
                          "Model default emotion size is inconsistent",
                          {{"reported_count", network.GetEmotionsCount()},
                           {"default_count", default_emotion_.size()}});
      }
      for (std::size_t index = 0; index < default_emotion_.size(); ++index) {
        if (!std::isfinite(default_emotion_[index]) ||
            default_emotion_[index] < 0.0F ||
            default_emotion_[index] > 1.0F) {
          throw WorkerError(
              "model_invalid",
              "Audio2Face default emotion is outside [0, 1]",
              {{"index", index}, {"value", default_emotion_[index]}});
        }
      }
      copy_emotion_names(network);

      diffusion_blendshape_info_ = require_sdk_ptr(
          nva2f::ReadDiffusionBlendshapeSolveModelInfo(
              request.audio2face_model_path.c_str()),
          "Reading diffusion blendshape solver", "model_invalid");
      const auto execution_option =
          nva2f::IGeometryExecutor::ExecutionOption::Skin |
          nva2f::IGeometryExecutor::ExecutionOption::Eyes;
      const auto blendshape_parameters =
          diffusion_blendshape_info_->GetExecutorCreationParameters(
              execution_option,
              static_cast<std::size_t>(request.identity_index));
      const auto skin_names =
          pose_names(blendshape_parameters.initializationSkinParams, "skin");
      arkit_indices_ = arkit52_indices(skin_names);

      bundle_ = require_sdk_ptr(
          nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
              1, request.audio2face_model_path.c_str(),
              execution_option, true,
              static_cast<std::size_t>(request.identity_index), true, nullptr,
              nullptr),
          "Creating diffusion GPU blendshape executor", "gpu_error");
      auto& executor = bundle_->GetExecutor();
      if (executor.GetResultType() !=
          nva2f::IBlendshapeExecutor::ResultsType::DEVICE) {
        throw WorkerError("gpu_error",
                          "Audio2Face did not create a device blendshape solver");
      }
      if (executor.GetNbTracks() != 1 ||
          executor.GetWeightCount() != kArkit52SdkNames.size()) {
        throw WorkerError(
            "model_invalid", "SDK blendshape layout does not match model metadata",
            {{"tracks", executor.GetNbTracks()},
             {"reported_weights", executor.GetWeightCount()},
             {"expected_weights", kArkit52SdkNames.size()}});
      }
      sdk_check(nva2f::GetExecutorGeometryExecutor(executor, &geometry_view_),
                "Retrieving geometry executor");
      if (geometry_view_ == nullptr) {
        throw WorkerError("sdk_error", "Geometry executor is null");
      }
      if (geometry_view_->GetEyesRotationSize() != kEyesRotationCount) {
        throw WorkerError("model_invalid", "Unsupported SDK eyes rotation size",
                          {{"reported", geometry_view_->GetEyesRotationSize()},
                           {"expected", kEyesRotationCount}});
      }
      sdk_check(nva2f::SetExecutorGeometryResultsCallback(
                    executor, &Impl::geometry_callback, this),
                "Installing geometry callback");
      sdk_check(executor.SetResultsCallback(&Impl::weights_callback, this),
                "Installing device blendshape callback");

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
      audio2emotion_classifier_count_ =
          emotion_model_info->GetNetworkInfo().GetEmotionsCount();
      if (audio2emotion_classifier_count_ == 0) {
        throw WorkerError("model_invalid",
                          "Audio2Emotion classifier has no output emotions");
      }
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
      audio2emotion_input_window_samples_ =
          classifier_parameters.networkInfo.bufferLength;
      if (audio2emotion_input_window_samples_ == 0 ||
          classifier_parameters.networkInfo.bufferSamplerate != sample_rate_) {
        throw WorkerError(
            "model_invalid",
            "Audio2Emotion reported invalid audio window metadata",
            {{"buffer_samples", audio2emotion_input_window_samples_},
             {"audio2emotion_sample_rate",
              classifier_parameters.networkInfo.bufferSamplerate},
             {"audio2face_sample_rate", sample_rate_}});
      }
      emotion_executor_ = require_sdk_ptr(
          nva2e::CreateClassifierEmotionExecutor(emotion_parameters,
                                                 classifier_parameters),
          "Creating Audio2Emotion GPU executor", "gpu_error");
      // SDK 1.0.0 exposes the post-processed vector width, but not names for
      // those output positions. The release catalog therefore pins one model
      // pair whose positional compatibility must be verified end to end.
      if (emotion_executor_->GetNbTracks() != 1 ||
          emotion_executor_->GetSamplingRate() != sample_rate_ ||
          emotion_executor_->GetEmotionsSize() != default_emotion_.size()) {
        throw WorkerError(
            "model_invalid",
            "Audio2Emotion output is incompatible with Audio2Face",
            {{"audio2face_sample_rate", sample_rate_},
             {"audio2emotion_sample_rate",
              emotion_executor_->GetSamplingRate()},
             {"audio2face_emotion_count", default_emotion_.size()},
             {"audio2emotion_output_count",
              emotion_executor_->GetEmotionsSize()}});
      }
      sdk_check(emotion_executor_->SetResultsCallback(
                    &Impl::emotion_callback, this),
                "Installing Audio2Emotion callback");

      float input_strength = 0.0F;
      nva2f::AnimatorSkinParams skin{};
      nva2e::PostProcessParams emotion_defaults{};
      sdk_check(nva2f::GetExecutorInputStrength(executor, input_strength),
                "Reading model input strength");
      sdk_check(nva2f::GetExecutorSkinParameters(executor, 0, skin),
                "Reading model skin parameter defaults");
      sdk_check(nva2e::GetExecutorPostProcessParameters(
                    *emotion_executor_, 0, emotion_defaults),
                "Reading Audio2Emotion parameter defaults");
      const std::size_t default_max_emotions =
          emotion_defaults.maxEmotions == 0
              ? audio2emotion_classifier_count_
              : emotion_defaults.maxEmotions;
      if (!std::isfinite(emotion_defaults.emotionStrength) ||
          !std::isfinite(emotion_defaults.emotionContrast) ||
          !std::isfinite(emotion_defaults.liveBlendCoef) ||
          !std::isfinite(emotion_defaults.liveTransitionTime) ||
          emotion_defaults.emotionStrength < 0.0F ||
          emotion_defaults.emotionStrength > 1.0F ||
          emotion_defaults.emotionContrast < 0.1F ||
          emotion_defaults.emotionContrast > 3.0F ||
          emotion_defaults.liveBlendCoef < 0.0F ||
          emotion_defaults.liveBlendCoef > 1.0F ||
          emotion_defaults.liveTransitionTime < 0.1F ||
          emotion_defaults.liveTransitionTime > 1.0F ||
          default_max_emotions == 0 ||
          default_max_emotions > audio2emotion_classifier_count_) {
        throw WorkerError(
            "model_invalid",
            "Audio2Emotion model has invalid post-process defaults");
      }
      json manual_values = json::object();
      for (std::size_t index = 0; index < emotion_names_.size(); ++index) {
        manual_values[emotion_names_[index]] = default_emotion_[index];
      }
      json automatic_emotion_defaults = {
          {"strength", emotion_defaults.emotionStrength},
          {"contrast", emotion_defaults.emotionContrast},
          {"smoothing", emotion_defaults.liveBlendCoef},
          {"transition_time", emotion_defaults.liveTransitionTime},
          {"max_emotions", default_max_emotions}};
      json parameter_defaults = {
          {"input_strength", input_strength},
          {"skin", skin_json(skin)},
          {"emotion",
           {{"manual_values", std::move(manual_values)},
            {"auto", std::move(automatic_emotion_defaults)}}}};
      return {{"parameter_defaults", std::move(parameter_defaults)},
              {"emotion_names", emotion_names_},
              {"sample_rate", sample_rate_}};
    } catch (...) {
      clear_locked();
      throw;
    }
  }

  void generate(const GenerateRequest& request, std::atomic_bool& canceled,
                const ProgressCallback& progress,
                const ResultPublicationGate& publication_gate) {
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Generation was canceled");
    }
    progress(0.01, "loading_audio");
    std::uint32_t sample_rate = 0;
    {
      std::lock_guard<std::mutex> lock(resource_mutex_);
      require_model_locked();
      sample_rate = sample_rate_;
    }
    const std::vector<float> audio =
        read_wav_mono(request.audio_path, sample_rate);
    if (audio.empty()) {
      throw WorkerError("audio_invalid", "Audio contains no samples");
    }
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Generation was canceled");
    }

    begin_operation(request.settings);
    OperationReset reset(*this);
    json timestamps = json::array();
    json weights = json::array();
    auto& timestamp_rows = timestamps.get_ref<json::array_t&>();
    auto& weight_rows = weights.get_ref<json::array_t&>();
    const auto collect = [&](const StreamFrame& frame) {
      if (weight_rows.size() >=
          kMaximumResultScalars / kArkit52SdkNames.size()) {
        throw WorkerError("result_too_large",
                          "Animation result exceeds the worker safety limit");
      }
      timestamp_rows.push_back(frame.timestamp_sample);
      weight_rows.push_back(frame.weights);
    };

    progress(0.05, "generating");
    for (std::size_t offset = 0; offset < audio.size();
         offset += kAudioChunkSamples) {
      if (canceled.load(std::memory_order_acquire)) {
        throw WorkerError("canceled", "Generation was canceled");
      }
      const std::size_t count =
          std::min(kAudioChunkSamples, audio.size() - offset);
      accumulate_audio(audio.data() + offset, count);
      drain_ready(canceled, collect);
      progress(0.05 + 0.85 * static_cast<double>(offset + count) /
                          static_cast<double>(audio.size()),
               "generating");
    }
    close_audio_and_drain(canceled, collect);
    const std::size_t expected_frames = executor().GetTotalNbFrames(0);
    if (weight_rows.empty() || weight_rows.size() != expected_frames) {
      throw WorkerError("generation_failed",
                        "SDK returned an incomplete frame set",
                        {{"expected", expected_frames},
                         {"received", weight_rows.size()}});
    }

    json document = {{"schema", "a2f-animation/1"},
                     {"job_id", request.job_id},
                     {"sample_rate", sample_rate},
                     {"timestamps_samples", std::move(timestamps)},
                     {"weights", std::move(weights)}};
    progress(0.97, "writing_result");
    write_json_atomically(request.result_path, document, canceled,
                          publication_gate);
  }

  json stream_start(const StreamRequest& request) {
    {
      std::lock_guard<std::mutex> lock(resource_mutex_);
      require_model_locked();
      if (request.sample_rate != sample_rate_) {
        throw WorkerError("sample_rate_mismatch",
                          "Streaming PCM must use the model sample rate",
                          {{"expected", sample_rate_},
                           {"received", request.sample_rate}});
      }
    }
    begin_operation(request.settings);
    try {
      active_stream_id_ = request.stream_id;
      const std::size_t prebuffer_samples =
          auto_audio2emotion_active_
              ? std::max(audio2face_input_window_samples_,
                         audio2emotion_input_window_samples_)
              : audio2face_input_window_samples_;
      return {{"sample_rate", sample_rate_},
              {"prebuffer_samples", prebuffer_samples}};
    } catch (...) {
      finish_operation();
      throw;
    }
  }

  void stream_chunk(const std::string& stream_id,
                    const std::vector<float>& audio,
                    std::atomic_bool& canceled,
                    const StreamFrameCallback& frame) {
    require_active_stream(stream_id);
    if (audio.empty()) {
      throw WorkerError("invalid_params", "Streaming PCM chunk must not be empty");
    }
    if (audio.size() > sample_rate_) {
      throw WorkerError("invalid_params",
                        "Streaming PCM chunk exceeds one second",
                        {{"maximum_samples", sample_rate_}});
    }
    for (std::size_t index = 0; index < audio.size(); ++index) {
      if (!std::isfinite(audio[index])) {
        throw WorkerError("invalid_params", "Streaming PCM must be finite",
                          {{"sample_index", index}});
      }
    }
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Stream was stopped");
    }
    accumulate_audio(audio.data(), audio.size());
    drain_ready(canceled, frame);
  }

  void stream_end(const std::string& stream_id,
                  std::atomic_bool& canceled,
                  const StreamFrameCallback& frame) {
    require_active_stream(stream_id);
    OperationReset reset(*this);
    close_audio_and_drain(canceled, frame);
  }

  void stream_abort(const std::string& stream_id) noexcept {
    if (operation_active_.load(std::memory_order_acquire) &&
        active_stream_id_ == stream_id) {
      active_stream_id_.clear();
      operation_active_.store(false, std::memory_order_release);
    }
  }

  void cancel() noexcept {
    // The noninteractive device executor is stopped by returning false from
    // its result callbacks. The operation's shared atomic flag is checked by
    // those callbacks and between every Execute call.
  }

 private:
  class OperationReset final {
   public:
    explicit OperationReset(Impl& owner) noexcept : owner_(&owner) {}
    OperationReset(const OperationReset&) = delete;
    OperationReset& operator=(const OperationReset&) = delete;
    ~OperationReset() {
      if (owner_ != nullptr) owner_->finish_operation();
    }

   private:
    Impl* owner_;
  };

  struct PendingFrame {
    std::int64_t next_timestamp{0};
    SdkPtr<nva2x::IHostTensorFloat> weights;
    SdkPtr<nva2x::IHostTensorFloat> eyes;
    bool has_weights{false};
    bool has_eyes{false};
  };

  struct Capture {
    std::atomic_bool* canceled{nullptr};
    cudaStream_t cuda_stream{nullptr};
    std::size_t weight_count{0};
    std::map<std::int64_t, PendingFrame> frames;
    bool failed{false};
    const char* failure{"SDK result callback failed"};
  };

  struct EmotionCapture {
    std::atomic_bool* canceled{nullptr};
    cudaStream_t cuda_stream{nullptr};
    nva2x::IEmotionAccumulator* accumulator{nullptr};
    std::size_t emotion_count{0};
    std::error_code accumulation_error;
    const char* failure{nullptr};
  };

  static void fail_capture(Capture& capture, const char* message) noexcept {
    capture.failed = true;
    capture.failure = message;
  }

  static bool emotion_callback(
      void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    auto& owner = *static_cast<Impl*>(userdata);
    if (owner.active_emotion_capture_ == nullptr) return false;
    auto& capture = *owner.active_emotion_capture_;
    if (capture.canceled->load(std::memory_order_acquire)) return false;
    if (results.trackIndex != 0 ||
        results.cudaStream != capture.cuda_stream ||
        results.emotions.Size() != capture.emotion_count) {
      capture.failure = "Audio2Emotion callback returned an invalid result";
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
      if (results.trackIndex != 0 ||
          results.eyesRotation.Size() != kEyesRotationCount ||
          results.eyesCudaStream != capture.cuda_stream) {
        fail_capture(capture, "Geometry callback returned an invalid result");
        return false;
      }
      auto& frame = capture.frames[results.timeStampCurrentFrame];
      if (frame.has_eyes ||
          (frame.has_weights &&
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
      frame.has_eyes = true;
      return true;
    } catch (const std::exception& error) {
      (void)error;
      fail_capture(capture, "Geometry callback failed");
      return false;
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
      if (results.trackIndex != 0 ||
          results.weights.Size() != capture.weight_count ||
          results.cudaStream != capture.cuda_stream) {
        fail_capture(capture,
                     "Blendshape callback returned an invalid result");
        return false;
      }
      auto& frame = capture.frames[results.timeStampCurrentFrame];
      if (frame.has_weights ||
          (frame.has_eyes && frame.next_timestamp != results.timeStampNextFrame)) {
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
      frame.has_weights = true;
      return true;
    } catch (const std::exception& error) {
      (void)error;
      fail_capture(capture, "Blendshape callback failed");
      return false;
    } catch (...) {
      fail_capture(capture, "Blendshape callback failed");
      return false;
    }
  }

  nva2f::IBlendshapeExecutor& executor() { return bundle_->GetExecutor(); }

  void require_model_locked() const {
    if (bundle_ == nullptr) {
      throw WorkerError("model_not_loaded", "Load a model before inference");
    }
  }

  void begin_operation(const json& settings) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    require_model_locked();
    if (operation_active_.exchange(true, std::memory_order_acq_rel)) {
      throw WorkerError("busy", "An Audio2Face operation is already active");
    }
    try {
      sdk_check(executor().Wait(0), "Waiting for prior blendshape work",
                "gpu_error");
      sdk_check(bundle_->GetCudaStream().Synchronize(),
                "Synchronizing CUDA stream", "gpu_error");
      sdk_check(executor().Reset(0), "Resetting blendshape executor");
      sdk_check(bundle_->GetAudioAccumulator(0).Reset(),
                "Resetting audio accumulator");
      sdk_check(bundle_->GetEmotionAccumulator(0).Reset(),
                "Resetting emotion accumulator");
      apply_settings(settings);
      if (auto_audio2emotion_active_) {
        sdk_check(emotion_executor_->Reset(0),
                  "Resetting Audio2Emotion executor");
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
      previous_timestamp_.reset();
    } catch (...) {
      operation_active_.store(false, std::memory_order_release);
      throw;
    }
  }

  void finish_operation() noexcept {
    active_stream_id_.clear();
    operation_active_.store(false, std::memory_order_release);
  }

  void require_active_stream(const std::string& stream_id) const {
    if (!operation_active_.load(std::memory_order_acquire) ||
        active_stream_id_ != stream_id) {
      throw WorkerError("job_not_found", "The requested stream is not active",
                        {{"job_id", stream_id}});
    }
  }

  void accumulate_audio(const float* audio, std::size_t count) {
    sdk_check(bundle_->GetAudioAccumulator(0).Accumulate(
                  nva2x::HostTensorFloatConstView(audio, count),
                  bundle_->GetCudaStream().Data()),
              "Accumulating audio");
  }

  void close_audio_and_drain(std::atomic_bool& canceled,
                             const StreamFrameCallback& frame) {
    sdk_check(bundle_->GetAudioAccumulator(0).Close(),
              "Closing audio accumulator");
    drain_interleaved_ready(canceled, frame);
    if (auto_audio2emotion_active_) {
      if (emotion_executor_->GetNbAvailableExecutions(0) != 0) {
        throw WorkerError(
            "generation_failed",
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
    capture.cuda_stream = bundle_->GetCudaStream().Data();
    capture.accumulator = &bundle_->GetEmotionAccumulator(0);
    capture.emotion_count = default_emotion_.size();
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
      throw WorkerError("generation_failed", capture.failure,
                        std::move(details));
    }
    sdk_check(execute_error, "Executing Audio2Emotion", "generation_failed");
  }

  void execute_face_once(std::atomic_bool& canceled,
                         const StreamFrameCallback& frame_callback) {
    if (canceled.load(std::memory_order_acquire)) {
      throw WorkerError("canceled", "Operation was stopped");
    }
    Capture capture;
    capture.canceled = &canceled;
    capture.cuda_stream = bundle_->GetCudaStream().Data();
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
    if (capture.failed) {
      throw WorkerError("generation_failed", capture.failure);
    }
    sdk_check(execute_error, "Executing Audio2Face", "generation_failed");

    for (const auto& [timestamp, pending] : capture.frames) {
      if (!pending.has_weights || !pending.has_eyes) {
        throw WorkerError("generation_failed",
                          "SDK callbacks returned an incomplete frame",
                          {{"timestamp", timestamp}});
      }
      if (previous_timestamp_.has_value() &&
          timestamp <= *previous_timestamp_) {
        throw WorkerError("generation_failed",
                          "SDK frame timestamps are not strictly increasing",
                          {{"timestamp", timestamp},
                           {"previous", *previous_timestamp_}});
      }
      std::vector<float> arkit;
      arkit.reserve(arkit_indices_.size());
      for (std::size_t channel = 0; channel < arkit_indices_.size(); ++channel) {
        const std::size_t sdk_channel = arkit_indices_[channel];
        const float value = pending.weights->Data()[sdk_channel];
        if (!std::isfinite(value)) {
          throw WorkerError(
              "generation_failed",
              "SDK produced a non-finite blendshape weight",
              {{"timestamp", timestamp}, {"channel", channel}});
        }
        arkit.push_back(value);
      }
      for (std::size_t index = 0; index < pending.eyes->Size(); ++index) {
        if (!std::isfinite(pending.eyes->Data()[index])) {
          throw WorkerError("generation_failed",
                            "SDK produced a non-finite eye rotation",
                            {{"timestamp", timestamp}, {"component", index}});
        }
      }
      resolve_arkit_eye_look(arkit, pending.eyes->Data());
      for (float& value : arkit) {
        if (!std::isfinite(value)) {
          throw WorkerError("generation_failed",
                            "ARKit resolver produced a non-finite weight",
                            {{"timestamp", timestamp}});
        }
        value = std::clamp(value, 0.0F, 1.0F);
      }
      if (canceled.load(std::memory_order_acquire)) {
        throw WorkerError("canceled", "Operation was stopped");
      }
      previous_timestamp_ = timestamp;
      frame_callback(StreamFrame{timestamp, std::move(arkit)});
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

  void apply_settings(const json& settings) {
    if (!settings.is_object()) {
      throw WorkerError("invalid_params", "settings must be an object");
    }
    require_exact_keys(settings, {"input_strength", "skin", "emotion"},
                       "settings");
    const auto skin_value = settings.find("skin");
    if (!skin_value->is_object()) {
      throw WorkerError("invalid_params", "settings.skin must be an object");
    }
    require_exact_keys(
        *skin_value,
        {"lower_face_smoothing", "upper_face_smoothing",
         "lower_face_strength", "upper_face_strength", "face_mask_level",
         "face_mask_softness", "skin_strength", "blink_strength",
         "eyelid_open_offset", "lip_open_offset", "blink_offset"},
        "settings.skin");
    nva2f::AnimatorSkinParams skin{};
    skin.lowerFaceSmoothing = required_float(
        *skin_value, "lower_face_smoothing", "settings.skin.");
    skin.upperFaceSmoothing = required_float(
        *skin_value, "upper_face_smoothing", "settings.skin.");
    skin.lowerFaceStrength = required_float(
        *skin_value, "lower_face_strength", "settings.skin.");
    skin.upperFaceStrength = required_float(
        *skin_value, "upper_face_strength", "settings.skin.");
    skin.faceMaskLevel =
        required_float(*skin_value, "face_mask_level", "settings.skin.");
    skin.faceMaskSoftness = required_float(
        *skin_value, "face_mask_softness", "settings.skin.");
    skin.skinStrength =
        required_float(*skin_value, "skin_strength", "settings.skin.");
    skin.blinkStrength =
        required_float(*skin_value, "blink_strength", "settings.skin.");
    skin.eyelidOpenOffset = required_float(
        *skin_value, "eyelid_open_offset", "settings.skin.");
    skin.lipOpenOffset =
        required_float(*skin_value, "lip_open_offset", "settings.skin.");
    skin.blinkOffset =
        required_float(*skin_value, "blink_offset", "settings.skin.");

    const auto emotion_value = settings.find("emotion");
    if (!emotion_value->is_object()) {
      throw WorkerError("invalid_params", "settings.emotion must be an object");
    }
    require_exact_keys(*emotion_value,
                       {"auto_audio2emotion", "manual_values", "auto"},
                       "settings.emotion");
    const bool auto_audio2emotion = required_bool(
        *emotion_value, "auto_audio2emotion", "settings.emotion.");
    const auto manual_value = emotion_value->find("manual_values");
    if (!manual_value->is_object() ||
        manual_value->size() != emotion_names_.size()) {
      throw WorkerError(
          "invalid_params",
          "settings.emotion.manual_values must contain every model emotion");
    }
    std::vector<float> manual_emotion;
    manual_emotion.reserve(emotion_names_.size());
    for (const std::string& name : emotion_names_) {
      if (!manual_value->contains(name)) {
        throw WorkerError(
            "invalid_params",
            "settings.emotion.manual_values is missing a model emotion",
            {{"emotion", name}});
      }
      manual_emotion.push_back(required_float_range(
          *manual_value, name.c_str(), "settings.emotion.manual_values.",
          0.0F, 1.0F));
    }

    const auto auto_value = emotion_value->find("auto");
    if (!auto_value->is_object()) {
      throw WorkerError("invalid_params",
                        "settings.emotion.auto must be an object");
    }
    require_exact_keys(*auto_value,
                       {"strength", "contrast", "smoothing",
                        "transition_time", "max_emotions"},
                       "settings.emotion.auto");
    const float emotion_strength = required_float_range(
        *auto_value, "strength", "settings.emotion.auto.", 0.0F, 1.0F);
    const float emotion_contrast = required_float_range(
        *auto_value, "contrast", "settings.emotion.auto.", 0.1F, 3.0F);
    const float emotion_smoothing = required_float_range(
        *auto_value, "smoothing", "settings.emotion.auto.", 0.0F, 1.0F);
    const float emotion_transition_time = required_float_range(
        *auto_value, "transition_time", "settings.emotion.auto.", 0.1F,
        1.0F);
    const std::size_t max_emotions = required_size_range(
        *auto_value, "max_emotions", "settings.emotion.auto.", 1,
        audio2emotion_classifier_count_);

    sdk_check(nva2f::SetExecutorInputStrength(
                  executor(),
                  required_float(settings, "input_strength", "settings.")),
              "Applying input strength");
    sdk_check(nva2f::SetExecutorSkinParameters(executor(), 0, skin),
              "Applying skin parameters");
    nva2e::PostProcessParams emotion_parameters{};
    sdk_check(nva2e::GetExecutorPostProcessParameters(
                  *emotion_executor_, 0, emotion_parameters),
              "Reading Audio2Emotion parameters");
    emotion_parameters.emotionStrength = emotion_strength;
    emotion_parameters.emotionContrast = emotion_contrast;
    emotion_parameters.liveBlendCoef = emotion_smoothing;
    emotion_parameters.liveTransitionTime = emotion_transition_time;
    emotion_parameters.maxEmotions = max_emotions;
    emotion_parameters.enablePreferredEmotion = false;
    sdk_check(nva2e::SetExecutorPostProcessParameters(
                  *emotion_executor_, 0, emotion_parameters),
              "Applying Audio2Emotion parameters");
    auto_audio2emotion_active_ = auto_audio2emotion;
    manual_emotion_ = std::move(manual_emotion);
  }

  void copy_default_emotion(nva2x::HostTensorFloatConstView emotion) {
    default_emotion_.clear();
    if (emotion.Size() != 0) {
      if (emotion.Data() == nullptr) {
        throw WorkerError("model_invalid", "Model default emotion view is null");
      }
      default_emotion_.assign(emotion.Data(), emotion.Data() + emotion.Size());
    }
  }

  template <class NetworkInfo>
  void copy_emotion_names(const NetworkInfo& network) {
    emotion_names_.clear();
    emotion_names_.reserve(network.GetEmotionsCount());
    for (std::size_t index = 0; index < network.GetEmotionsCount(); ++index) {
      const char* name = network.GetEmotionName(index);
      if (name == nullptr || *name == '\0') {
        throw WorkerError("model_invalid",
                          "Audio2Face contains an empty emotion name",
                          {{"index", index}});
      }
      if (std::find(emotion_names_.begin(), emotion_names_.end(), name) !=
          emotion_names_.end()) {
        throw WorkerError("model_invalid",
                          "Audio2Face contains duplicate emotion names",
                          {{"emotion", name}});
      }
      emotion_names_.emplace_back(name);
    }
  }

  void clear_locked() noexcept {
    if (bundle_ != nullptr) {
      (void)bundle_->GetExecutor().Wait(0);
      (void)bundle_->GetCudaStream().Synchronize();
    }
    geometry_view_ = nullptr;
    active_capture_ = nullptr;
    active_emotion_capture_ = nullptr;
    emotion_executor_.reset();
    bundle_.reset();
    diffusion_blendshape_info_.reset();
    diffusion_geometry_info_.reset();
    arkit_indices_.clear();
    emotion_names_.clear();
    default_emotion_.clear();
    manual_emotion_.clear();
    active_stream_id_.clear();
    previous_timestamp_.reset();
    sample_rate_ = 0;
    audio2emotion_classifier_count_ = 0;
    audio2face_input_window_samples_ = 0;
    audio2emotion_input_window_samples_ = 0;
    auto_audio2emotion_active_ = false;
    operation_active_.store(false, std::memory_order_release);
  }

  std::mutex resource_mutex_;
  std::atomic_bool operation_active_{false};
  SdkPtr<nva2f::IDiffusionModel::IGeometryModelInfo>
      diffusion_geometry_info_;
  SdkPtr<nva2f::IDiffusionModel::IBlendshapeSolveModelInfo>
      diffusion_blendshape_info_;
  SdkPtr<nva2f::IBlendshapeExecutorBundle> bundle_;
  SdkPtr<nva2e::IEmotionExecutor> emotion_executor_;
  nva2f::IGeometryExecutor* geometry_view_{nullptr};
  Capture* active_capture_{nullptr};
  EmotionCapture* active_emotion_capture_{nullptr};
  std::vector<std::size_t> arkit_indices_;
  std::vector<std::string> emotion_names_;
  std::vector<float> default_emotion_;
  std::vector<float> manual_emotion_;
  std::string active_stream_id_;
  std::optional<std::int64_t> previous_timestamp_;
  std::uint32_t sample_rate_{0};
  std::size_t audio2emotion_classifier_count_{0};
  std::size_t audio2face_input_window_samples_{0};
  std::size_t audio2emotion_input_window_samples_{0};
  bool auto_audio2emotion_active_{false};
};

Backend::Backend() : impl_(std::make_unique<Impl>()) {}
Backend::~Backend() = default;

json Backend::load_model(const ModelRequest& request) {
  return impl_->load_model(request);
}

void Backend::generate(const GenerateRequest& request,
                       std::atomic_bool& canceled,
                       const ProgressCallback& progress,
                       const ResultPublicationGate& publication_gate) {
  impl_->generate(request, canceled, progress, publication_gate);
}

json Backend::stream_start(const StreamRequest& request) {
  return impl_->stream_start(request);
}

void Backend::stream_chunk(const std::string& stream_id,
                           const std::vector<float>& audio,
                           std::atomic_bool& canceled,
                           const StreamFrameCallback& frame) {
  impl_->stream_chunk(stream_id, audio, canceled, frame);
}

void Backend::stream_end(const std::string& stream_id,
                         std::atomic_bool& canceled,
                         const StreamFrameCallback& frame) {
  impl_->stream_end(stream_id, canceled, frame);
}

void Backend::stream_abort(const std::string& stream_id) noexcept {
  impl_->stream_abort(stream_id);
}

void Backend::cancel() noexcept { impl_->cancel(); }

}  // namespace a2f_worker
