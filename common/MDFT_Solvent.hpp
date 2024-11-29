#ifndef MDFT_SOLVENT_HPP
#define MDFT_SOLVENT_HPP

#include <memory>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"

namespace MDFT {
// \brief Compute delta_rho(r, Omega) = rho0 * (xi(r, Omega) ^2 - 1)
// These arrays are stored in the same order, so we just recast this into 1D
// View and perform the operation with 1D parallel for loop \tparam
// ExecutionSpace Execution space \tparam ViewType View type \tparam ScalarType
// Scalar type
//
// \param exec_space [in] Execution space instance
// \param xi [in] 6D View of xi, shape(nx, ny, nz, ntheta, nphi, npsi)
// \param delta_rho [out] 6D View of delta_rho, shape(nx, ny, nz, ntheta, nphi,
// npsi) \param rho0 [in] Reference density
template <KokkosExecutionSpace ExecutionSpace, KokkosView ViewType,
          typename ScalarType>
  requires KokkosViewAccesible<ExecutionSpace, ViewType>
void get_delta_rho(const ExecutionSpace& exec_space, const ViewType& xi,
                   const ViewType& delta_rho, const ScalarType rho0) {
  const std::size_t n = xi.size();
  MDFT::Impl::Throw_If(delta_rho.size() != n,
                       "size of delta_rho must be the same as the size of xi");

  // Flatten Views for simplicity
  using ValueType  = typename ViewType::non_const_value_type;
  using View1DType = Kokkos::View<ValueType*, ExecutionSpace>;
  View1DType xi_1d(xi.data(), n), delta_rho_1d(delta_rho.data(), n);

  Kokkos::parallel_for(
      "delta_rho",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, n),
      KOKKOS_LAMBDA(std::size_t i) {
        delta_rho_1d(i) = rho0 * (xi_1d(i) * xi_1d(i) - 1.0);
      });
}

};  // namespace MDFT

#endif
