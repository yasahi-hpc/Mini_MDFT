#ifndef MDFT_GRID_HPP
#define MDFT_GRID_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include "MDFT_Asserts.hpp"

namespace MDFT {
/// \brief A class that manages Spatial Grid
///
/// \tparam ScalarType The type of the scalar values.
///
template <typename ScalarType>
struct SpatialGrid {
  using IntType         = int;
  using IntArrayType    = Kokkos::Array<IntType, 3>;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;
  using View1DType =
      typename Kokkos::View<ScalarType*, Kokkos::DefaultExecutionSpace>;

  // Spatial grid
  // number of grid nodes in direction x, y and z
  IntArrayType m_n_nodes, m_n;
  IntType m_nx, m_ny, m_nz;

  // total length in direction x, y and z
  ScalarArrayType m_length, m_l;
  ScalarType m_lx, m_ly, m_lz;

  // elemental distance between two nodes in direction x, y and z
  ScalarArrayType m_dl;
  ScalarType m_dx, m_dy, m_dz;
  ScalarType m_dv;  // elemental volume
  ScalarType m_v;
  ScalarType m_buffer_length;  // length of free space between the extremam of
                               // the solute.

  View1DType m_kx, m_ky, m_kz;

  SpatialGrid(IntArrayType n_nodes, ScalarArrayType length)
      : m_n_nodes(n_nodes), m_length(length) {
    MDFT::Impl::Throw_If(
        std::any_of(Kokkos::begin(m_length), Kokkos::end(m_length),
                    [](ScalarType value) { return value <= 0; }),
        "The supercell cannot have negative length.");
    MDFT::Impl::Throw_If(
        std::any_of(Kokkos::begin(m_n_nodes), Kokkos::end(m_n_nodes),
                    [](ScalarType value) { return value <= 0; }),
        "The space is divided into grid nodes.");
    m_nx = m_n_nodes[0], m_ny = m_n_nodes[1], m_nz = m_n_nodes[2];
    m_lx = m_length[0], m_ly = m_length[1], m_lz = m_length[2];
    // std::tie(m_nx, m_ny, m_nz) = m_n_nodes;
    // std::tie(m_lx, m_ly, m_lz) = m_length;

    std::transform(
        Kokkos::begin(m_length), Kokkos::end(m_length),
        Kokkos::begin(m_n_nodes), Kokkos::begin(m_dl),
        [](ScalarType l, IntType n) { return l / static_cast<ScalarType>(n); });

    m_v  = std::accumulate(Kokkos::begin(m_length), Kokkos::end(m_length), 1.0,
                           std::multiplies<ScalarType>());
    m_dv = std::accumulate(Kokkos::begin(m_dl), Kokkos::end(m_dl), 1.0,
                           std::multiplies<ScalarType>());
  }
};

template <typename ScalarType>
struct AngularGrid {
  using IntType         = int;
  using IntArrayType    = std::array<IntType, 3>;
  using ScalarArrayType = std::array<ScalarType, 3>;
  using View1DType =
      typename Kokkos::View<ScalarType*, Kokkos::DefaultExecutionSpace>;
  using View3DType =
      typename Kokkos::View<ScalarType***, Kokkos::DefaultExecutionSpace>;

  // Angular grid .. angular quadrature
  IntType m_molrotsymorder, m_mmax, m_ntheta, m_nphi, m_npsi, m_no, m_np;
  ScalarArrayType m_dphi, m_dpsi;

  View1DType m_theta, m_phi, m_psi, m_wtheta, m_wphi, m_wpsi, m_w;
  View1DType m_thetaofntheta, m_wthetaofntheta;
  View1DType m_phiofnphi, m_psiofnpsi, m_wphiofnphi, m_wpsiofnpsi;
  View3DType m_indo;  // table of index of orientations
  View3DType m_io;
  View1DType m_rotxx, m_rotxy, m_rotxz, m_rotyx, m_rotyy, m_rotyz, m_rotzx,
      m_rotzy, m_rotzz;
  View1DType m_OMx, m_OMy, m_OMz;

  AngularGrid(int mmax, int molrotsymorder)
      : m_mmax(mmax), m_molrotsymorder(molrotsymorder) {
    MDFT::Impl::Throw_If(mmax <= 0, "The grid mmax must be greater than zero.");
    MDFT::Impl::Throw_If(molrotsymorder <= 0,
                         "The grid molrotsymorder must be greater than zero.");
  }
};
}  // namespace MDFT

#endif  // MDFT_GRID_HPP
