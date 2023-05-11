project(linesplitter)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

FetchContent_Declare(
  nanoarrow
  URL https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-0.1.0/apache-arrow-nanoarrow-0.1.0.tar.gz
  URL_HASH SHA512=dc62480b986ee76aaad8e38c6fbc602f8cef2cc35a5f5ede7da2a93b4db2b63839bdca3eefe8a44ae1cb6895a2fd3f090e3f6ea1020cf93cfe86437304dfee17)
FetchContent_MakeAvailable(nanoarrow)

add_library(linesplitter linesplitter.cc)
target_link_libraries(linesplitter PRIVATE nanoarrow)
