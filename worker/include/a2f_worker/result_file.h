#pragma once

#include <atomic>
#include <functional>
#include <string>

#include <nlohmann/json.hpp>

namespace a2f_worker {

using ResultCommit = std::function<void()>;
using ResultPublicationGate =
    std::function<void(const ResultCommit& commit)>;

void write_json_atomically(const std::string& path,
                           const nlohmann::json& value,
                           const std::atomic_bool& canceled,
                           const ResultPublicationGate& publication_gate);

}  // namespace a2f_worker
