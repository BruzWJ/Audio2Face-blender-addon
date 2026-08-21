#include "wav.h"

#include "backend.h"
#include "path_contract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace a2f_worker {
namespace {

constexpr std::uint64_t kMaximumWavBytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumAudioSamples =
    kMaximumWavBytes / sizeof(float);
constexpr std::uint16_t kMaximumChannels = 32;
constexpr std::uint32_t kMinimumSampleRate = 8000;
constexpr std::uint32_t kMaximumSampleRate = 384000;
constexpr std::uint64_t kMaximumDurationSeconds = 6ULL * 60ULL * 60ULL;
constexpr unsigned char kPcmSubformatGuid[] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
constexpr unsigned char kFloatSubformatGuid[] = {
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};

std::uint16_t u16(const unsigned char* p) {
  return static_cast<std::uint16_t>(p[0]) |
         (static_cast<std::uint16_t>(p[1]) << 8U);
}

std::uint32_t u32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8U) |
         (static_cast<std::uint32_t>(p[2]) << 16U) |
         (static_cast<std::uint32_t>(p[3]) << 24U);
}

double decode_sample(const unsigned char* p, std::uint16_t format,
                     std::uint16_t bits) {
  if (format == 3 && bits == 32) {
    float value = 0.0F;
    std::memcpy(&value, p, sizeof(value));
    if (!std::isfinite(value)) {
      throw WorkerError("invalid_audio", "Float32 WAV contains a non-finite sample");
    }
    return std::clamp(static_cast<double>(value), -1.0, 1.0);
  }
  if (format != 1) {
    throw WorkerError("unsupported_audio", "Only PCM and float32 WAV files are supported");
  }
  switch (bits) {
    case 8:
      return (static_cast<double>(*p) - 128.0) / 128.0;
    case 16: {
      const auto raw = static_cast<std::int16_t>(u16(p));
      return static_cast<double>(raw) / 32768.0;
    }
    case 24: {
      std::int32_t raw = static_cast<std::int32_t>(p[0]) |
                         (static_cast<std::int32_t>(p[1]) << 8) |
                         (static_cast<std::int32_t>(p[2]) << 16);
      if ((raw & 0x00800000) != 0) raw |= static_cast<std::int32_t>(0xFF000000);
      return static_cast<double>(raw) / 8388608.0;
    }
    case 32: {
      const auto raw = static_cast<std::int32_t>(u32(p));
      return static_cast<double>(raw) / 2147483648.0;
    }
    default:
      throw WorkerError("unsupported_audio", "Unsupported PCM WAV bit depth",
                        {{"bits_per_sample", bits}});
  }
}

std::vector<float> resample_linear(const std::vector<float>& input,
                                   std::uint32_t source_rate,
                                   std::uint32_t target_rate) {
  // read_wav_mono bounds input.size() to kMaximumAudioSamples. Since both rates
  // are uint32, this multiplication cannot overflow uint64.
  const std::uint64_t scaled_samples =
      static_cast<std::uint64_t>(input.size()) * target_rate;
  const std::uint64_t requested_samples =
      std::max<std::uint64_t>(1, scaled_samples / source_rate);
  if (requested_samples > kMaximumAudioSamples) {
    throw WorkerError(
        "audio_too_large",
        "Resampled WAV exceeds the 512 MiB decoded-audio safety limit",
        {{"source_samples", input.size()},
         {"source_sample_rate", source_rate},
         {"target_sample_rate", target_rate},
         {"requested_samples", requested_samples},
         {"maximum_samples", kMaximumAudioSamples}});
  }
  if (source_rate == target_rate) return input;
  const auto output_size = static_cast<std::size_t>(requested_samples);
  std::vector<float> output(output_size);
  for (std::size_t i = 0; i < output_size; ++i) {
    const std::uint64_t numerator =
        static_cast<std::uint64_t>(i) * source_rate;
    const auto left = std::min(
        static_cast<std::size_t>(numerator / target_rate), input.size() - 1);
    const auto right = std::min(left + 1, input.size() - 1);
    const double fraction =
        static_cast<double>(numerator % target_rate) / target_rate;
    const double left_value = input[left];
    const double right_value = input[right];
    output[i] = static_cast<float>(
        left_value + (right_value - left_value) * fraction);
  }
  return output;
}

}  // namespace

std::vector<float> read_wav_mono(const std::string& path,
                                 std::uint32_t target_sample_rate) {
  const auto source_path = require_canonical_regular_file(
      path, "audio_open_failed", "WAV path");
  std::ifstream stream(source_path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw WorkerError("audio_open_failed", "Could not open WAV file", {{"path", path}});
  }
  const auto end = stream.tellg();
  if (end < 12 || static_cast<std::uint64_t>(end) > kMaximumWavBytes) {
    throw WorkerError("invalid_audio", "WAV file size is invalid", {{"path", path}});
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!stream || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw WorkerError("invalid_audio", "File is not a valid RIFF/WAVE file", {{"path", path}});
  }

  const std::uint64_t riff_end = static_cast<std::uint64_t>(u32(bytes.data() + 4)) + 8ULL;
  if (riff_end != bytes.size()) {
    throw WorkerError("invalid_audio", "RIFF size does not match the WAV file size",
                      {{"path", path}});
  }

  const unsigned char* fmt = nullptr;
  std::size_t fmt_size = 0;
  const unsigned char* data = nullptr;
  std::size_t data_size = 0;
  for (std::size_t offset = 12; offset < bytes.size();) {
    if (bytes.size() - offset < 8) {
      throw WorkerError("invalid_audio", "WAV contains a truncated chunk header",
                        {{"path", path}});
    }
    const std::uint32_t chunk_size = u32(bytes.data() + offset + 4);
    const std::size_t payload = offset + 8;
    if (payload > bytes.size() || chunk_size > bytes.size() - payload) {
      throw WorkerError("invalid_audio", "WAV contains a truncated chunk", {{"path", path}});
    }
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0) {
      if (fmt != nullptr) {
        throw WorkerError("invalid_audio", "WAV contains more than one fmt chunk",
                          {{"path", path}});
      }
      fmt = bytes.data() + payload;
      fmt_size = chunk_size;
    } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0) {
      if (data != nullptr) {
        throw WorkerError("invalid_audio", "WAV contains more than one data chunk",
                          {{"path", path}});
      }
      data = bytes.data() + payload;
      data_size = chunk_size;
    }
    const std::size_t padded = static_cast<std::size_t>(chunk_size) + (chunk_size & 1U);
    if (padded > bytes.size() - payload) {
      throw WorkerError("invalid_audio", "WAV chunk extends beyond the RIFF container",
                        {{"path", path}});
    }
    offset = payload + padded;
  }
  if (fmt == nullptr || fmt_size < 16 || data == nullptr) {
    throw WorkerError("invalid_audio", "WAV is missing fmt or data chunks", {{"path", path}});
  }

  std::uint16_t format = u16(fmt);
  const std::uint16_t channels = u16(fmt + 2);
  const std::uint32_t source_rate = u32(fmt + 4);
  const std::uint32_t byte_rate = u32(fmt + 8);
  const std::uint16_t block_align = u16(fmt + 12);
  const std::uint16_t bits = u16(fmt + 14);
  if (format == 0xFFFE) {
    if (fmt_size < 40) {
      throw WorkerError("invalid_audio", "Extensible WAV fmt chunk is shorter than 40 bytes");
    }
    const std::uint16_t extension_size = u16(fmt + 16);
    const std::uint16_t valid_bits = u16(fmt + 18);
    if (extension_size < 22 ||
        static_cast<std::size_t>(extension_size) + 18U > fmt_size ||
        valid_bits != bits) {
      throw WorkerError("invalid_audio", "Extensible WAV format fields are invalid");
    }
    if (std::memcmp(fmt + 24, kPcmSubformatGuid,
                    sizeof(kPcmSubformatGuid)) == 0) {
      format = 1;
    } else if (std::memcmp(fmt + 24, kFloatSubformatGuid,
                           sizeof(kFloatSubformatGuid)) == 0) {
      format = 3;
    } else {
      throw WorkerError("unsupported_audio", "Extensible WAV subformat is unsupported");
    }
  } else if (format != 1 && format != 3) {
    throw WorkerError("unsupported_audio", "WAV must contain PCM or float32 audio");
  }
  if (channels < 1 || channels > kMaximumChannels ||
      source_rate < kMinimumSampleRate || source_rate > kMaximumSampleRate ||
      target_sample_rate < kMinimumSampleRate ||
      target_sample_rate > kMaximumSampleRate || bits % 8U != 0) {
    throw WorkerError("invalid_audio", "WAV format fields are outside supported bounds",
                      {{"path", path}});
  }
  const std::size_t bytes_per_sample = bits / 8U;
  if ((format == 3 && bits != 32) ||
      (format == 1 && (bytes_per_sample < 1 || bytes_per_sample > 4))) {
    throw WorkerError("unsupported_audio", "WAV sample width is unsupported");
  }
  if (block_align != channels * bytes_per_sample ||
      byte_rate != static_cast<std::uint64_t>(source_rate) * block_align ||
      data_size == 0 || data_size % block_align != 0) {
    throw WorkerError("invalid_audio", "WAV format fields are inconsistent", {{"path", path}});
  }

  const std::size_t frames = data_size / block_align;
  if (frames > static_cast<std::uint64_t>(source_rate) *
                   kMaximumDurationSeconds) {
    throw WorkerError("audio_too_large", "WAV duration exceeds six hours",
                      {{"path", path}});
  }
  const std::uint64_t output_frames = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(frames) * target_sample_rate / source_rate);
  if (frames > kMaximumAudioSamples || output_frames > kMaximumAudioSamples) {
    throw WorkerError(
        "audio_too_large",
        "WAV exceeds the 512 MiB decoded-audio safety limit",
        {{"source_samples", frames},
         {"output_samples", output_frames},
         {"maximum_samples", kMaximumAudioSamples}});
  }
  std::vector<float> mono(frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    double sum = 0.0;
    for (std::uint16_t channel = 0; channel < channels; ++channel) {
      const auto* sample = data + frame * block_align + channel * bytes_per_sample;
      sum += decode_sample(sample, format, bits);
    }
    mono[frame] = static_cast<float>(sum / channels);
  }

  return resample_linear(mono, source_rate, target_sample_rate);
}

}  // namespace a2f_worker
