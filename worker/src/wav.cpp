#include "a2f_worker/wav.h"

#include "a2f_worker/backend.h"

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

float decode_sample(const unsigned char* p, std::uint16_t format,
                    std::uint16_t bits) {
  if (format == 3 && bits == 32) {
    float value = 0.0F;
    std::memcpy(&value, p, sizeof(value));
    if (!std::isfinite(value)) {
      throw WorkerError("invalid_audio", "Float32 WAV contains a non-finite sample");
    }
    return std::clamp(value, -1.0F, 1.0F);
  }
  if (format != 1) {
    throw WorkerError("unsupported_audio", "Only PCM and float32 WAV files are supported");
  }
  switch (bits) {
    case 8:
      return (static_cast<float>(*p) - 128.0F) / 128.0F;
    case 16: {
      const auto raw = static_cast<std::int16_t>(u16(p));
      return static_cast<float>(raw) / 32768.0F;
    }
    case 24: {
      std::int32_t raw = static_cast<std::int32_t>(p[0]) |
                         (static_cast<std::int32_t>(p[1]) << 8) |
                         (static_cast<std::int32_t>(p[2]) << 16);
      if ((raw & 0x00800000) != 0) raw |= static_cast<std::int32_t>(0xFF000000);
      return static_cast<float>(raw) / 8388608.0F;
    }
    case 32: {
      const auto raw = static_cast<std::int32_t>(u32(p));
      return static_cast<float>(raw) / 2147483648.0F;
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
  const double ratio = static_cast<double>(target_rate) / source_rate;
  std::vector<float> output(output_size);
  for (std::size_t i = 0; i < output_size; ++i) {
    const double position = static_cast<double>(i) / ratio;
    const auto left = std::min(static_cast<std::size_t>(position), input.size() - 1);
    const auto right = std::min(left + 1, input.size() - 1);
    const float fraction = static_cast<float>(position - static_cast<double>(left));
    output[i] = input[left] + (input[right] - input[left]) * fraction;
  }
  return output;
}

}  // namespace

std::vector<float> read_wav_mono(const std::string& path,
                                 std::uint32_t target_sample_rate) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw WorkerError("audio_open_failed", "Could not open WAV file", {{"path", path}});
  }
  const auto end = stream.tellg();
  if (end < 44 || static_cast<std::uint64_t>(end) > kMaximumWavBytes) {
    throw WorkerError("invalid_audio", "WAV file size is invalid", {{"path", path}});
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!stream || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw WorkerError("invalid_audio", "File is not a valid RIFF/WAVE file", {{"path", path}});
  }

  const unsigned char* fmt = nullptr;
  std::size_t fmt_size = 0;
  const unsigned char* data = nullptr;
  std::size_t data_size = 0;
  for (std::size_t offset = 12; offset + 8 <= bytes.size();) {
    const std::uint32_t chunk_size = u32(bytes.data() + offset + 4);
    const std::size_t payload = offset + 8;
    if (payload > bytes.size() || chunk_size > bytes.size() - payload) {
      throw WorkerError("invalid_audio", "WAV contains a truncated chunk", {{"path", path}});
    }
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0) {
      fmt = bytes.data() + payload;
      fmt_size = chunk_size;
    } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0 && data == nullptr) {
      data = bytes.data() + payload;
      data_size = chunk_size;
    }
    const std::size_t padded = static_cast<std::size_t>(chunk_size) + (chunk_size & 1U);
    if (padded > bytes.size() - payload) break;
    offset = payload + padded;
  }
  if (fmt == nullptr || fmt_size < 16 || data == nullptr) {
    throw WorkerError("invalid_audio", "WAV is missing fmt or data chunks", {{"path", path}});
  }

  std::uint16_t format = u16(fmt);
  const std::uint16_t channels = u16(fmt + 2);
  const std::uint32_t source_rate = u32(fmt + 4);
  const std::uint16_t block_align = u16(fmt + 12);
  const std::uint16_t bits = u16(fmt + 14);
  if (format == 0xFFFE && fmt_size >= 40) format = u16(fmt + 24);  // extensible subformat
  const std::size_t bytes_per_sample = (bits + 7U) / 8U;
  if (channels == 0 || source_rate == 0 || target_sample_rate == 0 ||
      bytes_per_sample == 0 || block_align == 0 ||
      block_align != channels * bytes_per_sample || data_size % block_align != 0) {
    throw WorkerError("invalid_audio", "WAV format fields are inconsistent", {{"path", path}});
  }

  const std::size_t frames = data_size / block_align;
  if (frames == 0) throw WorkerError("invalid_audio", "WAV has no audio samples");
  if (frames > kMaximumAudioSamples) {
    throw WorkerError(
        "audio_too_large",
        "WAV exceeds the 512 MiB decoded-audio safety limit",
        {{"source_samples", frames}, {"maximum_samples", kMaximumAudioSamples}});
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
