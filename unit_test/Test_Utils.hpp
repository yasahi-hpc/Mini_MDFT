// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include "MDFT_Concepts.hpp"

using namespace MDFT;

// namespace MDFT {
template <KokkosExecutionSpace ExecutionSpace, KokkosView AViewType,
          KokkosView BViewType>
bool allclose(const ExecutionSpace& exec, const AViewType& a,
              const BViewType& b, double rtol = 1.e-5, double atol = 1.e-8) {
  constexpr std::size_t rank = AViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      "MDFT::Test::allclose",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const int& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  return error == 0;
}

inline bool is_included(const std::string& str,
                        const std::vector<std::string>& vec) {
  return std::find(vec.begin(), vec.end(), str) != vec.end();
}

//}  // namespace MDFT
#endif
