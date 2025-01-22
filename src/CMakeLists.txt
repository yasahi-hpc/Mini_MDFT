# SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_executable(mini-MDFT mini-mdft.cpp)

target_compile_features(mini-MDFT PUBLIC cxx_std_20)
target_link_libraries(mini-MDFT PUBLIC Mini-MDFT-Common)
