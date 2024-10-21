#ifndef MDFT_MATH_UTILS_HPP
#define MDFT_MATH_UTILS_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include "MDFT_Asserts.hpp"
#include "MDFT_Concepts.hpp"

namespace MDFT {
namespace Impl {
/// \brief Compute Gauss-Legendre quadrature.
///        Compute sampling points and weights for Gauss-Legendre quadrature
///
/// \tparam InViewType: Input type, needs to be a 1D View
/// \tparam OutViewType: Output type, needs to be a 1D View
///
template <KokkosView InViewType, KokkosView OutViewType>
void gauss_legendre(InViewType& x, OutViewType& w) {
  using value_type = typename InViewType::non_const_value_type;
  auto h_x         = Kokkos::create_mirror_view(x);
  auto h_w         = Kokkos::create_mirror_view(w);

  int n = x.extent(0);
  MDFT::Impl::Throw_If(w.extent(0) != n,
                       "Input and output should have the same extent");
  int m = (n + 1) / 2;

  for (int i = 1; i <= m; ++i) {
    // Initial approximation of the root
    value_type xi      = Kokkos::cos(M_PI * (i - 0.25) / (n + 0.5));
    value_type deltaxi = 1.0;

    // Newton's method to refine the root
    value_type pp = 0.0;
    while (Kokkos::abs(deltaxi) >
           std::numeric_limits<value_type>::epsilon()) {  // epsilon is around
                                                          // 1e-15
      value_type p1 = 1.0;
      value_type p2 = 0.0;

      // Recurrence relation to compute Legendre polynomial values
      for (int j = 1; j <= n; ++j) {
        value_type p3 = p2;
        p2            = p1;
        p1            = ((2 * j - 1) * xi * p2 - (j - 1) * p3) / j;
      }

      // Derivative of the Legendre polynomial
      pp = n * (xi * p1 - p2) / (xi * xi - 1.0);

      // Newton's method update
      deltaxi = -p1 / pp;
      xi += deltaxi;
    }

    // Store root and weight
    h_x(i - 1) = xi;
    h_w(i - 1) = 1.0 / ((1.0 - xi * xi) * pp * pp);

    // Symmetric roots and weights
    h_x(n - i) = -xi;
    h_w(n - i) = h_w(i - 1);
  }
  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(w, h_w);
}

template <KokkosExecutionSpace ExecutionSpace, KokkosView ViewType>
requires KokkosViewAccesible<ExecutionSpace, ViewType>
void uniform_mesh(const ExecutionSpace& exec, ViewType& x,
                  typename ViewType::non_const_value_type dx) {
  using value_type = typename ViewType::non_const_value_type;
  Kokkos::parallel_for(
      "uniform_mesh",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec, 0, x.size()),
      KOKKOS_LAMBDA(const int i) { x(i) = static_cast<value_type>(i) * dx; });
  exec.fence();
}

}  // namespace Impl
}  // namespace MDFT

#endif
