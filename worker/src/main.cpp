#include "protocol.h"

#include <cstdio>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main() {
#ifdef _WIN32
  // The JSONL wire contract is LF-only. The MSVC CRT otherwise translates
  // stdout LF to CRLF and silently removes CR from stdin in text mode.
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
      _setmode(_fileno(stdout), _O_BINARY) == -1) {
    std::fputs("Fatal worker error: could not configure binary JSONL pipes\n",
               stderr);
    return 1;
  }
#endif
  return a2f_worker::run_protocol_server();
}
