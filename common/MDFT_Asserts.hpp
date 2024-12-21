#ifndef MDFT_ASSERTS_HPP
#define MDFT_ASSERTS_HPP

#include <stdexcept>
#include <sstream>
#include <string_view>
#include <source_location>

namespace MDFT {
namespace Impl {
inline void Throw_If(
    const bool expression, const std::string_view& msg,
    const char* file_name     = std::source_location::current().file_name(),
    int line                  = std::source_location::current().line(),
    const char* function_name = std::source_location::current().function_name(),
    const int column          = std::source_location::current().column()) {
  // Quick return if possible
  if (!expression) return;

  std::stringstream ss("file: ");
  ss << file_name << '(' << line << ':' << column << ") `" << function_name
     << "`: " << msg << '\n';
  throw std::runtime_error(ss.str());
}
}  // namespace Impl
}  // namespace MDFT

#endif
