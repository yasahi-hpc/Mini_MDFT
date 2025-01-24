// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}

