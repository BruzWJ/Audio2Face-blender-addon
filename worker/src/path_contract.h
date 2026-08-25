#pragma once

#include <string>

namespace a2f_worker {

void require_canonical_regular_file(const std::string& value,
                                    const char* error_code,
                                    const char* label);

}  // namespace a2f_worker
