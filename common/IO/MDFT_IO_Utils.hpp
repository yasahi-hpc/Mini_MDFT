#ifndef MDFT_IO_UTILS_HPP
#define MDFT_IO_UTILS_HPP

#include <string_view>
#include <fstream>
#include <iostream>

namespace MDFT {
namespace IO {
namespace Impl {
inline bool is_file_exists(const std::string file_name) {
  std::ifstream file(file_name);
  return file.good();
}

inline int count_lines_in_file(const std::string& filename) {
  std::ifstream file(filename);

  if (file) {
    std::string line;
    int lineCount = 0;
    while (std::getline(file, line)) {
      ++lineCount;
    }
    return lineCount;
  } else {
    std::cerr << "Error: Could not open the file." << std::endl;
    return -1;  // Error code
  }
}

}  // namespace Impl
}  // namespace IO
}  // namespace MDFT

#endif
