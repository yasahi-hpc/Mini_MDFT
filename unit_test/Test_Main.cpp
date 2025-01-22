// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize();
  auto result = RUN_ALL_TESTS();
  Kokkos::finalize();

  return result;
}
