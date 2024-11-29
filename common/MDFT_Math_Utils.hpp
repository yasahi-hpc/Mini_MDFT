#ifndef MDFT_MATH_UTILS_HPP
#define MDFT_MATH_UTILS_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <KokkosFFT.hpp>
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

// \brief Compute the cross product of two 3D vectors
template <KokkosArray ArrayType>
KOKKOS_INLINE_FUNCTION auto cross_product(const ArrayType a,
                                          const ArrayType b) {
  ArrayType result;
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
  return result;
}

// \brief Compute the Euclidean vector norm (L2 norm)
template <KokkosArray ArrayType>
KOKKOS_INLINE_FUNCTION auto norm2(const ArrayType x) ->
    typename ArrayType::value_type {
  using value_type  = typename ArrayType::value_type;
  value_type l2nrom = 0;
  for (int i = 0; i < x.size(); ++i) {
    l2nrom += x[i] * x[i];
  }

  return Kokkos::sqrt(l2nrom);
}

template <KokkosArray ArrayType>
KOKKOS_INLINE_FUNCTION auto L2normalize(const ArrayType x) {
  ArrayType result;
  auto tmp_norm2 = norm2(x);
  for (int i = 0; i < x.size(); ++i) {
    result[i] = x[i] / tmp_norm2;
  }
  return result;
}

// Corresponds to ix_mq, iy_mq, iz_mq in Fortran
template <typename IntType>
KOKKOS_INLINE_FUNCTION auto inv_index(const IntType i, const IntType n) {
  IntType result = i == 0 ? 0 : n - i + 1;
  return result;
}

template <typename ValueType, typename RealType>
KOKKOS_INLINE_FUNCTION ValueType prevent_underflow(const ValueType x,
                                                   const RealType epsilon) {
  ValueType result = x;
  if constexpr (KokkosFFT::Impl::is_complex_v<ValueType>) {
    if (Kokkos::abs(result.real()) < epsilon) result.real() = 0.0;
    if (Kokkos::abs(result.imag()) < epsilon) result.imag() = 0.0;
  } else {
    if (Kokkos::abs(result) < epsilon) result = 0.0;
  }
  return result;
}

}  // namespace Impl
}  // namespace MDFT

#endif
