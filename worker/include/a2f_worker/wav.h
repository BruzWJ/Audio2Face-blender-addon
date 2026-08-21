#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace a2f_worker {

std::vector<float> read_wav_mono(const std::string& path,
                                 std::uint32_t target_sample_rate);

}  // namespace a2f_worker
