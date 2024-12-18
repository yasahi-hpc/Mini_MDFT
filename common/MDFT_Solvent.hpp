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
// View and perform the operation with 1D parallel for loop
// \tparam ExecutionSpace Execution space
// \tparam ViewType View type
// \tparam ScalarType Scalar type
//
// \param exec_space [in] Execution space instance
// \param xi [in] 6D View of xi, shape(nx, ny, nz, ntheta, nphi, npsi)
// \param delta_rho [out] 6D View of delta_rho, shape(nx, ny, nz, ntheta, nphi,
// npsi)
// \param rho0 [in] Reference density
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

// \brief Gather projections into gamma
// \tparam ExecutionSpace Execution space
// \tparam ViewType View type
// \tparam ScalarType Scalar type
//
// \param exec_space [in] Execution space instance
// \param xi [in] 6D View of xi, shape(nx, ny, nz, ntheta, nphi, npsi)
// \param vexc [in] 3D View of vexc, shape(ntheta, nphi, npsi)
// \param w [in] 3D View of w, shape(ntheta, nphi, npsi) (volume element in
// angular space?)
// \param delta_f [out] 6D View of delta_f, shape(nx, ny, nz,
// ntheta, nphi, npsi)
// \param ff [out] density fluctuation
// \param rho0 [in] Reference density
// \param prefactor [in] Coefficient prefactor
template <KokkosExecutionSpace ExecutionSpace, KokkosView View6DType,
          KokkosView View3DType, typename ScalarType>
  requires KokkosViewAccesible<ExecutionSpace, View3DType> &&
           KokkosViewAccesible<ExecutionSpace, View6DType>
void get_delta_f(const ExecutionSpace& exec_space, const View6DType& xi,
                 const View3DType& vexc, const View3DType& w,
                 const View6DType& delta_f, ScalarType& ff,
                 const ScalarType rho0, const ScalarType prefactor) {
  const std::size_t nx = xi.extent(0), ny = xi.extent(1), nz = xi.extent(2),
                    ntheta = xi.extent(3), nphi = xi.extent(4),
                    npsi = xi.extent(5);

  for (int i = 0; i < 3; i++) {
    MDFT::Impl::Throw_If(xi.extent(i + 3) != vexc.extent(i),
                         "angular grid size of xi and vexc must be the same");
  }

  // Flatten Views for simplicity
  const std::size_t nxyz = nx * ny * nz, nangle = ntheta * nphi * npsi;
  using ValueType  = typename View6DType::non_const_value_type;
  using View1DType = Kokkos::View<ValueType*, ExecutionSpace>;
  using View2DType = Kokkos::View<ValueType**, ExecutionSpace>;
  View1DType vexc_1d(vexc.data(), nangle), w_1d(w.data(), nangle);
  View2DType delta_f_2d(delta_f.data(), nxyz, nangle),
      xi_2d(xi.data(), nxyz, nangle);

  ff                = 0;
  using member_type = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  auto team_policy  = Kokkos::TeamPolicy<ExecutionSpace>(nxyz, Kokkos::AUTO);
  Kokkos::parallel_reduce(
      "delta_f", team_policy,
      KOKKOS_LAMBDA(const member_type& team_member, ValueType& l_ff) {
        const auto ixyz = team_member.league_rank();
        ValueType sum   = 0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team_member, nangle),
            [&](const int ip, ValueType& lsum) {
              lsum += vexc_1d(ip) * w_1d(ip) *
                      (xi_2d(ixyz, ip) * xi_2d(ixyz, ip) - 1.0);
              delta_f_2d(ixyz, ip) = 2.0 * rho0 * xi_2d(ixyz, ip) * vexc_1d(ip);
            },
            sum);
        l_ff += rho0 * sum * prefactor;
      },
      ff);
}

};  // namespace MDFT

#endif
