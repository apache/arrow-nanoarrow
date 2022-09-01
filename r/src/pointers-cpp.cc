
#include <string>
#include <cstring>

extern "C" void intptr_as_string(intptr_t ptr_int, char* buf) {
  std::string ptr_str = std::to_string(ptr_int);
  memcpy(buf, ptr_str.data(), ptr_str.size());
}
