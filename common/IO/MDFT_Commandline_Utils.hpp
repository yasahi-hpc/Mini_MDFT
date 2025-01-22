// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#ifndef MDFT_COMMANDLINE_UTILS_HPP
#define MDFT_COMMANDLINE_UTILS_HPP

#include <map>
#include <vector>
#include <string>
#include <cassert>
#include "MDFT_String_Utils.hpp"

namespace MDFT {
namespace IO {

using dict = std::map<std::string, std::string>;

dict parse_args(int argc, char* argv[]) {
  dict kwargs;
  const std::vector<std::string> args(argv + 1, argv + argc);

  assert(args.size() % 2 == 0);

  for (std::size_t i = 0; i < args.size(); i += 2) {
    std::string key   = Impl::trimLeft(args[i], "-");
    std::string value = args[i + 1];
    kwargs[key]       = value;
  }
  return kwargs;
}

std::string get_arg(dict& kwargs, const std::string& key,
                    const std::string& default_value = "") {
  if (kwargs.find(key) != kwargs.end()) {
    return kwargs[key];
  } else {
    return default_value;
  }
}

}  // namespace IO
}  // namespace MDFT

#endif
